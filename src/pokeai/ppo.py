"""PPO トレーナー（ロードマップ Phase 1 手順3・5）。CleanRL 風の1ファイル完結実装。

    uv run python -m pokeai.ppo --config configs/ppo_smoke.yaml

各実行は ``runs/<name>-<timestamp>/`` に、解決済みの設定・TensorBoard ログ・
``checkpoints/latest.pt``＋定期スナップショット・self-play プールを書き出す。

対戦相手のカリキュラムは ``{until_step, mix}`` のステージのリストで、``mix`` は
対戦相手の名前（``OpponentFactory`` 参照）を抽選比重に対応付けたもの。``latest`` /
``pool`` は自分自身のスナップショットを指す、つまり self-play（最新 50% + 過去 50% など）。
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from pokeai import encoding
from pokeai.env import EnvConfig, make_env
from pokeai.model import N_ACTIONS, ActorCritic, ModelConfig, load_checkpoint, save_checkpoint


@dataclass
class Stage:
    until_step: Optional[int]  # None = 学習終了まで継続
    mix: dict


@dataclass
class _PendingTP:
    """エピソードの結末を待っているチームプレビューの選出（TEAMPREVIEW_PICK 回分の
    サブ選出）。選出が行われた時点からエピソード終了まで、報酬を（割引しながら）
    積算していき、そのエピソードの全選出に対する学習ターゲットとなる
    モンテカルロ収益を得る。"""

    obs: np.ndarray
    mask: np.ndarray
    action: np.ndarray
    logprob: np.ndarray
    value: np.ndarray
    ret: float = 0.0
    disc: float = 1.0


@dataclass
class TrainConfig:
    run_name: str = "ppo"
    seed: int = 1
    device: str = "auto"
    total_timesteps: int = 1_000_000
    num_envs: int = 8
    num_steps: int = 256
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    checkpoint_every_updates: int = 20
    snapshot_every_updates: int = 20  # 凍結したコピーを self-play プールに追加する頻度
    pool_size: int = 20
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    curriculum: list[Stage] = field(default_factory=lambda: [Stage(None, {"random": 1.0})])

    # YAML ファイルから TrainConfig を読み込む。書かれていない項目は既定値のまま。
    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        env = EnvConfig(**raw.pop("env", {}))
        model = ModelConfig(**raw.pop("model", {}))
        stages = [Stage(s.get("until_step"), s["mix"]) for s in raw.pop("curriculum", [])]
        cfg = cls(**raw, env=env, model=model)
        if stages:
            cfg.curriculum = stages
        return cfg

    # 現在の学習ステップに対応するカリキュラムの対戦相手比率を返す。
    def mix_at(self, step: int) -> dict:
        for s in self.curriculum:
            if s.until_step is None or step < s.until_step:
                return s.mix
        return self.curriculum[-1].mix


# デバイスを決定する。CUDA があればそれを、無ければ CPU を使う
# （ネットワークが小さく、ロールアウト中のバッチも小さいので Mac では MPS より CPU の方が速く、
#  自動選択では MPS は選ばない）。
def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PPO 学習のメインループ: ロールアウト収集 → GAE 計算 → ミニバッチ更新 →
# ログ出力・チェックポイント保存、を num_updates 回繰り返す。
def train(cfg: TrainConfig, resume: Optional[str] = None) -> Path:
    run_dir = Path("runs") / f"{cfg.run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    ckpt_dir, pool_dir = run_dir / "checkpoints", run_dir / "pool"
    ckpt_dir.mkdir(parents=True)
    pool_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
    writer = SummaryWriter(str(run_dir / "tb"))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    batch_size = cfg.num_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = max(1, cfg.total_timesteps // batch_size)

    env_cfg = dataclasses.replace(cfg.env, opponent_mix=cfg.mix_at(0))
    envs = gym.vector.AsyncVectorEnv(
        [make_env(env_cfg, i) for i in range(cfg.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        context="spawn",
    )

    if resume:
        agent, ckpt = load_checkpoint(resume, device)
        agent.train()
    else:
        agent, ckpt = ActorCritic(cfg.model).to(device), {}
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    global_step = int(ckpt.get("global_step", 0))

    obs_dim = envs.single_observation_space["observation"].shape[0]
    obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim), device=device)
    mask_buf = torch.zeros((cfg.num_steps, cfg.num_envs, N_ACTIONS), device=device)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)

    pool: list[str] = []
    current_mix: dict = cfg.mix_at(global_step)
    win_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
    turns_hist: deque = deque(maxlen=200)
    ep_returns = np.zeros(cfg.num_envs)
    return_hist: deque = deque(maxlen=200)

    # チームプレビューの選出（env.PokemonEnv._agent1_teampreview）はエピソードに
    # つき1回、通常の観測/行動/報酬のステップループの外で行われるので、ここで
    # 別途トラッキングし、そのエピソードが終わった時点でモンテカルロ収益の
    # 遷移として PPO のバッチに合流させる（下記参照）。
    pending_tp: dict[int, _PendingTP] = {}
    tp_ready: list[_PendingTP] = []

    # 新しく行われたチームプレビューの選出（infos["teampreview"]）を、
    # 環境インデックスをキーに pending_tp へ取り込む。
    def ingest_teampreview(infos: dict) -> None:
        present = infos.get("_teampreview")
        batch = infos.get("teampreview")
        if present is None or batch is None:
            return
        for i in np.flatnonzero(present):
            pending_tp[int(i)] = _PendingTP(
                obs=batch["obs"][i], mask=batch["mask"][i], action=batch["action"][i],
                logprob=batch["logprob"][i], value=batch["value"][i],
            )

    # 現在のモデルを凍結して self-play プールに追加し、全環境へ配布する
    # （対戦相手としても、チームプレビュー用ポリシーとしても使われる）。
    def snapshot() -> None:
        path = pool_dir / f"step_{global_step:09d}.pt"
        save_checkpoint(path, agent, global_step=global_step)
        pool.append(str(path.resolve()))
        del pool[: -cfg.pool_size]
        envs.call("set_opponent_mix", current_mix, pool)
        envs.call("reload_teampreview", str(path.resolve()))

    # モデル・オプティマイザ状態・ステップ数・学習設定一式を `path` に保存する。
    def save(path: Path) -> None:
        save_checkpoint(
            path, agent, optimizer_state=optimizer.state_dict(),
            global_step=global_step, train_config=dataclasses.asdict(cfg),
        )

    snapshot()  # プールを空にしないことで "latest"/"pool" が必ず解決できるようにする
    next_obs, next_infos = envs.reset(seed=cfg.seed)
    ingest_teampreview(next_infos)
    next_o = torch.as_tensor(next_obs["observation"], device=device)
    next_m = torch.as_tensor(next_obs["action_mask"], dtype=torch.float32, device=device)
    next_done = torch.zeros(cfg.num_envs, device=device)
    start = time.time()
    start_step = global_step

    for update in range(1, num_updates + 1):
        mix = cfg.mix_at(global_step)
        if mix != current_mix:
            current_mix = mix
            envs.call("set_opponent_mix", current_mix, pool)
            print(f"[step {global_step}] opponent mix -> {current_mix}")
        if cfg.anneal_lr:
            optimizer.param_groups[0]["lr"] = cfg.learning_rate * (1.0 - (update - 1.0) / num_updates)

        agent.eval()
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs_buf[step], mask_buf[step], dones[step] = next_o, next_m, next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_o, next_m)
            values[step], actions[step], logprobs[step] = value, action, logprob

            obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_o = torch.as_tensor(obs["observation"], device=device)
            next_m = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

            for i, tp in pending_tp.items():
                tp.ret += tp.disc * float(reward[i])
                tp.disc *= cfg.gamma

            ep_returns += reward
            final = infos.get("final_info", {})
            for i in np.flatnonzero(done):
                return_hist.append(ep_returns[i])
                ep_returns[i] = 0.0
                if final.get("_battle_won", np.zeros(cfg.num_envs, bool))[i]:
                    win_hist[str(final["opponent"][i])].append(final["battle_won"][i])
                    turns_hist.append(final["battle_turns"][i])
                if int(i) in pending_tp:
                    tp_ready.append(pending_tp.pop(int(i)))
            ingest_teampreview(infos)

        # GAE（Generalized Advantage Estimation）の計算
        with torch.no_grad():
            next_value = agent.get_value(next_o, next_m)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal, nextvalues = 1.0 - next_done, next_value
                else:
                    nextnonterminal, nextvalues = 1.0 - dones[t + 1], values[t + 1]
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_mask = mask_buf.reshape(-1, N_ACTIONS)
        b_actions, b_logprobs = actions.reshape(-1), logprobs.reshape(-1)
        b_adv, b_returns, b_values = advantages.reshape(-1), returns.reshape(-1), values.reshape(-1)

        # チームプレビューの選出: 今回の更新のロールアウト中に終了したエピソードは、
        # モンテカルロ収益の遷移（アドバンテージ = 収益 − 価値、つまりこのステップだけ
        # λ=1 の GAE 相当）として同じバッチに合流させ、交代ヘッド/価値ヘッドは
        # 変更せずそのまま使う。
        n_tp = len(tp_ready) * encoding.TEAMPREVIEW_PICK
        if tp_ready:
            tp_obs = torch.as_tensor(np.concatenate([tp.obs for tp in tp_ready]), device=device)
            tp_mask = torch.as_tensor(np.concatenate([tp.mask for tp in tp_ready]), device=device)
            tp_actions = torch.as_tensor(np.concatenate([tp.action for tp in tp_ready]), device=device)
            tp_logprobs = torch.as_tensor(np.concatenate([tp.logprob for tp in tp_ready]), device=device)
            tp_values = torch.as_tensor(np.concatenate([tp.value for tp in tp_ready]), device=device)
            tp_returns = torch.as_tensor(
                np.repeat([tp.ret for tp in tp_ready], encoding.TEAMPREVIEW_PICK), dtype=torch.float32, device=device
            )
            b_obs = torch.cat([b_obs, tp_obs])
            b_mask = torch.cat([b_mask, tp_mask])
            b_actions = torch.cat([b_actions, tp_actions])
            b_logprobs = torch.cat([b_logprobs, tp_logprobs])
            b_values = torch.cat([b_values, tp_values])
            b_returns = torch.cat([b_returns, tp_returns])
            b_adv = torch.cat([b_adv, tp_returns - tp_values])
            tp_ready.clear()

        agent.train()
        clipfracs = []
        total = b_obs.shape[0]
        b_inds = np.arange(total)
        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for s in range(0, total, minibatch_size):
                mb = b_inds[s : s + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_mask[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()

                if cfg.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(
                        newvalue - b_values[mb], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int((global_step - start_step) / (time.time() - start))

        w = writer.add_scalar
        w("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        w("losses/value_loss", v_loss.item(), global_step)
        w("losses/policy_loss", pg_loss.item(), global_step)
        w("losses/entropy", entropy_loss.item(), global_step)
        w("losses/approx_kl", approx_kl.item(), global_step)
        w("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        w("losses/explained_variance", explained_var, global_step)
        w("charts/SPS", sps, global_step)
        w("charts/teampreview_episodes", n_tp / encoding.TEAMPREVIEW_PICK, global_step)
        if return_hist:
            w("charts/episodic_return", float(np.mean(return_hist)), global_step)
        if turns_hist:
            w("charts/battle_turns", float(np.mean(turns_hist)), global_step)
        for opp, hist in win_hist.items():
            w(f"win_rate/{opp}", float(np.mean(hist)), global_step)

        rates = " ".join(f"{o}={np.mean(h):.2f}({len(h)})" for o, h in sorted(win_hist.items()))
        print(
            f"update {update}/{num_updates} step={global_step} sps={sps} "
            f"loss={loss.item():.3f} ent={entropy_loss.item():.3f} kl={approx_kl.item():.4f} "
            f"win[{rates}]",
            flush=True,
        )

        if update % cfg.snapshot_every_updates == 0:
            snapshot()
        if update % cfg.checkpoint_every_updates == 0:
            save(ckpt_dir / f"step_{global_step:09d}.pt")
        save(ckpt_dir / "latest.pt")

    envs.close()
    writer.close()
    print(f"done: {run_dir}")
    return run_dir


# CLI エントリポイント: 引数をパースして設定を読み込み、学習を実行する。
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/ppo_default.yaml")
    parser.add_argument("--resume", help="checkpoint to continue from (model + optimizer)")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--num-envs", type=int)
    args = parser.parse_args()
    cfg = TrainConfig.load(args.config)
    if args.total_timesteps:
        cfg.total_timesteps = args.total_timesteps
    if args.num_envs:
        cfg.num_envs = args.num_envs
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
