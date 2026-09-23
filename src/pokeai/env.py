"""Gymnasium environment around poke-env's SinglesEnv (roadmap Phase 1 step 1).

``PokemonEnv`` adds our observation encoding and (optionally shaped) reward.
``make_env`` builds a single-agent env whose opponent is re-sampled from an
opponent mix at every reset, which is what the PPO trainer runs in parallel
subprocesses. The opponent mix can be changed at runtime through
``AsyncVectorEnv.call("set_opponent_mix", ...)`` for curriculum / self-play.
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
    """Terminal ±victory, plus optional dense shaping (roadmap 0.3, ablation)."""

    victory: float = 1.0
    fainted: float = 0.0
    hp: float = 0.0
    status: float = 0.0

    # （空でもよい）辞書から RewardConfig を作る。
    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RewardConfig":
        return cls(**(d or {}))


class PokemonEnv(SinglesEnv):
    """poke-env's ``_EnvPlayer.teampreview`` only routes Team Preview through
    learned actions for VGC; for singles it always brings a random 3 of 6
    (see poke_env.environment.env._EnvPlayer._teampreview). We replace
    ``agent1``'s Team Preview with a synchronous model-driven pick (mirroring
    how ``PolicyPlayer`` already drives opponents synchronously) so it can be
    learned, and treat the picks as ordinary switch actions (0-5) restricted
    by ``teampreview_action_mask`` so no new model head is needed. The
    opponent side (``agent2``) is unaffected and keeps poke-env's default
    random Team Preview.
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
        """Refresh agent1's Team Preview policy from a checkpoint on disk
        (called periodically from the training process, like the self-play
        pool's checkpoints)."""
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
    """Single-agent view of PokemonEnv with an opponent sampled per episode.

    Exposes a flat ``Dict(observation, action_mask)`` space (as produced by
    PokeEnv) and reports the battle outcome in ``info`` on the final step.
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
    """Return a thunk for gymnasium vector envs (each runs in its own process)."""

    def _thunk() -> gym.Env:
        env = PokemonEnv(
            reward=RewardConfig.from_dict(cfg.reward),
            battle_format=cfg.battle_format,
            server_configuration=server_configuration(),
            account_configuration1=account(f"ppo{index}"),
            account_configuration2=account(f"opp{index}"),
            # Masked policies never pick illegal actions, but fall back to a
            # random legal move instead of crashing a long training run.
            strict=False,
            start_listening=True,
        )
        factory = OpponentFactory(cfg.battle_format, device=device)
        single = SingleAgentWrapper(env, factory.get("random"))
        return OpponentMixEnv(single, factory, cfg.opponent_mix)

    return _thunk
