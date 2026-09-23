"""対戦相手: poke-env のベースラインと凍結ポリシー（self-play プール）。

``PolicyPlayer`` はベンチマーク評価の対象そのものでもあるので、
学習時の対戦相手と評価対象のエージェントは同じ推論経路を共有する。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
from poke_env.player import (
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

from pokeai import encoding
from pokeai.model import ActorCritic, load_checkpoint
from pokeai.server import account

BASELINES = {
    "random": RandomPlayer,
    "max_power": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


@dataclass
class TeampreviewResult:
    """1回分のチームプレビュー選出: ``/team`` の指令文字列と、
    学習用の選出ごとの遷移（選ばれた自分のポケモンのスロット、順番通り）。"""

    order: str
    obs: np.ndarray  # (TEAMPREVIEW_PICK, OBS_DIM)
    mask: np.ndarray  # (TEAMPREVIEW_PICK, N_ACTIONS)
    action: np.ndarray  # (TEAMPREVIEW_PICK,) int64、自分のチーム内スロット番号
    logprob: np.ndarray  # (TEAMPREVIEW_PICK,)
    value: np.ndarray  # (TEAMPREVIEW_PICK,)


@torch.no_grad()
def run_teampreview(
    model: ActorCritic, battle: AbstractBattle, device="cpu", deterministic: bool = False
) -> TeampreviewResult:
    """モデルの交代ヘッドを使って、6匹のうち TEAMPREVIEW_PICK 匹を（先発順込みで）選ぶ。
    通常の交代アクション分布をそのまま流用し、まだ選んでいない自分のポケモンだけに
    マスクした上で TEAMPREVIEW_PICK 回繰り返す。次の回の前に選出済みのポケモンを
    ``_selected_in_teampreview`` としてマークし、マスク/観測を更新していく。"""
    team = list(battle.team.values())
    obs_list, mask_list, actions, logprobs, values = [], [], [], [], []
    picks: list[int] = []
    for _ in range(encoding.TEAMPREVIEW_PICK):
        obs = encoding.encode_battle(battle)
        mask = np.asarray(encoding.teampreview_action_mask(battle), dtype=np.float32)
        action, logprob, _, value = model.get_action_and_value(
            torch.as_tensor(obs, device=device).unsqueeze(0),
            torch.as_tensor(mask, device=device).unsqueeze(0),
            deterministic=deterministic,
        )
        idx = int(action.item())
        team[idx]._selected_in_teampreview = True
        picks.append(idx)
        obs_list.append(obs)
        mask_list.append(mask)
        actions.append(idx)
        logprobs.append(float(logprob.item()))
        values.append(float(value.item()))
    remaining = [i for i in range(len(team)) if i not in picks]
    order = "".join(str(i + 1) for i in picks + remaining)
    return TeampreviewResult(
        order=f"/team {order}",
        obs=np.stack(obs_list).astype(np.float32),
        mask=np.stack(mask_list).astype(np.float32),
        action=np.array(actions, dtype=np.int64),
        logprob=np.array(logprobs, dtype=np.float32),
        value=np.array(values, dtype=np.float32),
    )


class PolicyPlayer(Player):
    """ActorCritic で動く Player（choose_move が同期的で、env の対戦相手としても使える）。"""

    # 学習済みモデルで動く Player を作る（推論時間も計測する）。
    def __init__(self, model: ActorCritic, deterministic: bool = False, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.deterministic = deterministic
        self.device = device
        self.inference_seconds = 0.0
        self.n_decisions = 0

    # poke-env のフック: モデルで行動を選ぶ（合法手が無ければランダムな合法手にフォールバック）。
    @torch.no_grad()
    def choose_move(self, battle: AbstractBattle):
        mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=np.float32)
        if mask.sum() == 0:
            return self.choose_random_move(battle)
        start = time.perf_counter()
        obs = torch.as_tensor(encoding.encode_battle(battle), device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, device=self.device).unsqueeze(0)
        action, _, _, _ = self.model.get_action_and_value(obs, m, deterministic=self.deterministic)
        self.inference_seconds += time.perf_counter() - start
        self.n_decisions += 1
        return SinglesEnv.action_to_order(
            np.int64(action.item()), battle, fake=False, strict=False
        )

    # poke-env のフック: モデルでチームプレビューの選出を行う。
    def teampreview(self, battle: AbstractBattle) -> str:
        return run_teampreview(self.model, battle, self.device, self.deterministic).order


class OpponentFactory:
    """名前から対戦相手を生成（＆キャッシュ）する。

    名前: ``random`` / ``max_power`` / ``heuristic``（poke-env のベースライン）、
    ``latest``（最新の self-play スナップショット）、``pool``（self-play プールから
    エピソードごとに再抽選するスナップショット）、``ckpt:<path>``。
    """

    def __init__(self, battle_format: str, device: str = "cpu"):
        self.battle_format = battle_format
        self.device = device
        self._cache: dict[str, Player] = {}
        self.pool: list[str] = []

    # self-play プールを差し替え、外れたモデルのキャッシュを解放する。
    def set_pool(self, pool: list[str]) -> None:
        self.pool = list(pool)
        # プールから外れたモデルはキャッシュから外し、メモリ使用量を抑える。
        keep = {"ckpt:" + p for p in self.pool}
        for key in [k for k in self._cache if k.startswith("ckpt:") and k not in keep]:
            del self._cache[key]

    # サーバに接続しない Player を作る（環境内の相手は choose_move() だけ呼べればよい）。
    def _offline(self, cls, **kwargs) -> Player:
        return cls(
            account_configuration=account("offline"),
            battle_format=self.battle_format,
            start_listening=False,
            **kwargs,
        )

    # 名前から対戦相手を取得する（無ければ作ってキャッシュする）。
    def get(self, name: str) -> Player:
        if name in ("latest", "pool"):
            if not self.pool:
                return self.get("random")
            name = "ckpt:" + (self.pool[-1] if name == "latest" else random.choice(self.pool))
        if name not in self._cache:
            if name in BASELINES:
                self._cache[name] = self._offline(BASELINES[name])
            elif name.startswith("ckpt:"):
                model, _ = load_checkpoint(name[5:], self.device)
                self._cache[name] = self._offline(PolicyPlayer, model=model, device=self.device)
            else:
                raise ValueError(f"unknown opponent {name!r}")
        return self._cache[name]


# 評価・実戦投入用に、サーバへ接続する PolicyPlayer をチェックポイントから作る。
def make_policy_player(
    checkpoint: str,
    battle_format: str,
    deterministic: bool = False,
    device: str = "cpu",
    max_concurrent_battles: int = 1,
    server_configuration=None,
    name: Optional[str] = None,
    account_configuration=None,
    **kwargs,
) -> PolicyPlayer:
    model, _ = load_checkpoint(checkpoint, device)
    return PolicyPlayer(
        model=model,
        deterministic=deterministic,
        device=device,
        account_configuration=account_configuration or account(name or "agent"),
        battle_format=battle_format,
        server_configuration=server_configuration,
        max_concurrent_battles=max_concurrent_battles,
        **kwargs,
    )
