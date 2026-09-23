"""poke-env の SinglesEnv を包む Gymnasium 環境（ロードマップ Phase 1 手順1）。

``PokemonEnv`` はこちらの観測エンコーディングと（任意で途中報酬付きの）報酬計算を追加する。
``make_env`` は、リセットのたびに対戦相手をオッポーネントミックスから再抽選する
シングルエージェント環境を作る関数で、PPO トレーナーが並列サブプロセスで
実行するのはこれ。オッポーネントミックスは実行中でも
``AsyncVectorEnv.call("set_opponent_mix", ...)`` でカリキュラム/self-play 用に変更できる。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import Player

from pokeai import encoding
from pokeai.model import ActorCritic, load_checkpoint
from pokeai.opponents import OpponentFactory, run_teampreview
from pokeai.server import account, server_configuration


@dataclass
class RewardConfig:
    """終局の ±victory 報酬に、任意で密な途中報酬（shaping）を加える（ロードマップ 0.3、アブレーション実験）。"""

    victory: float = 1.0
    fainted: float = 0.0
    hp: float = 0.0
    status: float = 0.0

    # （空でもよい）辞書から RewardConfig を作る。
    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RewardConfig":
        return cls(**(d or {}))


class PokemonEnv(SinglesEnv):
    """poke-env の ``_EnvPlayer.teampreview`` は VGC でのみチームプレビューを
    学習対象の行動として扱い、シングルバトルでは常にランダムに6匹から3匹を選ぶ
    （poke_env.environment.env._EnvPlayer._teampreview 参照）。そこで ``agent1``
    のチームプレビューを、（``PolicyPlayer`` が既に対戦相手を同期的に動かしている
    のと同様の）同期的なモデル駆動の選出に差し替えて学習できるようにし、
    選出は通常の交代アクション（0〜5）を ``teampreview_action_mask`` で
    制限したものとして扱うので新しいモデルヘッドは不要。対戦相手側
    （``agent2``）はこの変更の影響を受けず、poke-env 既定のランダム選出のまま。
    """

    # 観測空間を設定し、自分側のチームプレビューをモデル駆動の版に差し替える。
    def __init__(self, *, reward: RewardConfig = RewardConfig(), **kwargs: Any):
        super().__init__(**kwargs)
        self.reward_cfg = reward
        space = Box(-np.inf, np.inf, shape=(encoding.OBS_DIM,), dtype=np.float32)
        self.observation_spaces = {agent: space for agent in self.possible_agents}
        self._teampreview_model: Optional[ActorCritic] = None
        self._teampreview_device = "cpu"
        self._pending_teampreview: Optional[dict] = None
        self.agent1.teampreview = self._agent1_teampreview

    # 自分側エージェントのチームプレビュー処理。モデルが読み込まれていれば
    # それで選出し、無ければ poke-env 既定のランダム選出にフォールバックする。
    def _agent1_teampreview(self, battle: AbstractBattle) -> str:
        if self._teampreview_model is None:
            return self.agent1.random_teampreview(battle)
        result = run_teampreview(
            self._teampreview_model, battle, self._teampreview_device, deterministic=False
        )
        self._pending_teampreview = {
            "obs": result.obs,
            "mask": result.mask,
            "action": result.action,
            "logprob": result.logprob,
            "value": result.value,
        }
        return result.order

    def reload_teampreview(self, path: str) -> None:
        """agent1 のチームプレビュー用ポリシーを、ディスク上のチェックポイントから
        再読み込みする（self-play プールのチェックポイントと同様、学習プロセスから
        定期的に呼ばれる）。"""
        model, _ = load_checkpoint(path, self._teampreview_device)
        self._teampreview_model = model

    # poke-env のフック: Battle を観測ベクトルに変換する。
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return encoding.encode_battle(battle)

    # poke-env のフック: RewardConfig に基づく（途中報酬込みの）報酬を計算する。
    def calc_reward(self, battle: AbstractBattle) -> float:
        r = self.reward_cfg
        return self.reward_computing_helper(
            battle,
            fainted_value=r.fainted,
            hp_value=r.hp,
            status_value=r.status,
            victory_value=r.victory,
        )

    # poke-env のフック: 行動 ID をバトルの指令に変換する。
    # チームプレビュー中は「交代アクション＝選出」として扱う。
    @staticmethod
    def action_to_order(action, battle: Battle, fake: bool = False, strict: bool = True):
        if battle.teampreview:
            return Player.create_order(list(battle.team.values())[int(action)])
        return SinglesEnv.action_to_order(action, battle, fake=fake, strict=strict)

    # poke-env のフック: 合法行動マスクを返す。チームプレビュー中は専用のマスクを使う。
    @staticmethod
    def get_action_mask(battle: Battle) -> list[int]:
        if battle.teampreview:
            return encoding.teampreview_action_mask(battle)
        return SinglesEnv.get_action_mask(battle)

    # バトルをリセットし、直前にチームプレビューの選出が行われていれば
    # 学習ループが拾えるよう infos に載せる。
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._pending_teampreview = None
        obs, infos = super().reset(seed, options)
        if self._pending_teampreview is not None:
            infos[self.agent1.username]["teampreview"] = self._pending_teampreview
        return obs, infos


class OpponentMixEnv(gym.Wrapper):
    """PokemonEnv をエピソードごとに対戦相手を抽選するシングルエージェント視点で包む。

    （PokeEnv が作る）フラットな ``Dict(observation, action_mask)`` 空間を公開し、
    最終ステップでは対戦結果を ``info`` に載せて返す。
    """

    def __init__(self, env: SingleAgentWrapper, factory: OpponentFactory, mix: dict):
        super().__init__(env)
        self.factory = factory
        self.mix: dict[str, float] = dict(mix)
        self.current_opponent = ""

    # 対戦相手の抽選比率（と self-play プール）を差し替える（学習ループから呼ばれる）。
    def set_opponent_mix(self, mix: dict, pool: Optional[list[str]] = None) -> None:
        self.mix = dict(mix)
        if pool is not None:
            self.factory.set_pool(pool)

    # チームプレビュー用モデルの再読み込みを、内側の PokemonEnv に転送する。
    def reload_teampreview(self, path: str) -> None:
        self.env.env.reload_teampreview(path)

    # `mix` の重みに従って対戦相手を1体抽選し、factory から取得/生成する。
    def _sample_opponent(self) -> None:
        names, weights = zip(*[(k, v) for k, v in self.mix.items() if v > 0])
        name = random.choices(names, weights=weights)[0]
        self.env.opponent = self.factory.get(name)
        self.current_opponent = name

    # 新しい対戦相手を抽選してから、内側の環境をリセットする。
    def reset(self, **kwargs):
        self._sample_opponent()
        obs, info = self.env.reset(**kwargs)
        info["opponent"] = self.current_opponent
        return obs, info

    # 内側の環境を1ステップ進め、試合終了時は勝敗などを info に付与する。
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(np.int64(action))
        info = dict(info)
        if terminated or truncated:
            battle = self.env.env.battle1
            info["battle_won"] = float(bool(battle.won))
            info["battle_turns"] = float(battle.turn)
            info["opponent"] = self.current_opponent
        return obs, reward, terminated, truncated, info


@dataclass
class EnvConfig:
    battle_format: str = encoding.DEFAULT_FORMAT
    reward: dict = field(default_factory=dict)
    opponent_mix: dict = field(default_factory=lambda: {"random": 1.0})


def make_env(cfg: EnvConfig, index: int = 0, device: str = "cpu"):
    """gymnasium のベクトル環境用のサンク関数を返す（各環境は独自のプロセスで動く）。"""

    def _thunk() -> gym.Env:
        env = PokemonEnv(
            reward=RewardConfig.from_dict(cfg.reward),
            battle_format=cfg.battle_format,
            server_configuration=server_configuration(),
            account_configuration1=account(f"ppo{index}"),
            account_configuration2=account(f"opp{index}"),
            # マスクされたポリシーが違法な行動を選ぶことは無いはずだが、
            # 万一来た場合は長時間の学習を落とさずランダムな合法手にフォールバックする。
            strict=False,
            start_listening=True,
        )
        factory = OpponentFactory(cfg.battle_format, device=device)
        single = SingleAgentWrapper(env, factory.get("random"))
        return OpponentMixEnv(single, factory, cfg.opponent_mix)

    return _thunk
