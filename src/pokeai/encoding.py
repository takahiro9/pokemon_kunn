"""Battle -> fixed-size observation vector (roadmap 0.3 観測エンコーディング).

Target format: [Gen 9 Champions] Random Battle (``gen9championsrandombattle``),
i.e. Pokemon Champions singles mechanics: Mega Evolution but no Terastal, and
move PP capped at 20 with Champions' own PP formula.

The observation is a flat float32 vector so it fits a gymnasium Box and can be
batched cheaply, but it is laid out in named blocks that ``split_obs`` turns
back into tensors for the model:

    [GLOBAL] [POKEMON_NUM x 12] [POKEMON_CAT x 12] [MOVE_NUM x 12 x 4]

Pokemon slots 0-5 are our team in ``battle.team`` order (which is the order
SinglesEnv uses for switch actions 0-5), slots 6-11 are the opponent Pokemon
revealed so far. Move slots follow ``pokemon.moves`` order (which is the order
SinglesEnv uses for move actions 6-9), so the model can point at the exact
move/switch an action refers to.

Categorical ids (species, item, ability, move) are stored as floats in the
vector and embedded by the model. Id 0 = padding / unknown.
"""

from __future__ import annotations

import zlib
from functools import lru_cache
from typing import Optional

import numpy as np
from poke_env.battle import (
    AbstractBattle,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.data import GenData

GEN = 9
DEFAULT_FORMAT = "gen9championsrandombattlelv50"
CHAMPIONS_PP_CAP = 20
N_TEAM = 6
N_SLOTS = 2 * N_TEAM
N_MOVES = 4
TEAMPREVIEW_PICK = 3  # Champions Flat Rules: bring 6, pick 3
N_ACTIONS = 26  # SinglesEnv.get_action_space_size(9): 6 switches + 4 moves x 5 (plain + 4 gimmicks)

TYPES = list(PokemonType)
STATUSES = list(Status)
WEATHERS = list(Weather)
FIELDS = list(Field)
SIDE_CONDITIONS = list(SideCondition)
CATEGORIES = list(MoveCategory)
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]
STACKABLE = {SideCondition.SPIKES: 3, SideCondition.TOXIC_SPIKES: 2}

HASH_BUCKETS = 1024  # items / abilities: poke-env ships no id table for them
UNKNOWN_ID = 0
NONE_ID = 1  # "known to have no item / ability"


# 種族名・技名 -> ID の対応表を作る（1 始まり。0 は不明/パディング用）。
@lru_cache(maxsize=1)
def _vocab() -> tuple[dict[str, int], dict[str, int]]:
    data = GenData.from_gen(GEN)
    # 0 = pad/unknown; real entries start at 1.
    species = {k: i + 1 for i, k in enumerate(sorted(data.pokedex))}
    moves = {k: i + 1 for i, k in enumerate(sorted(data.moves))}
    return species, moves


# 種族埋め込みの語彙数（不明用の ID 0 を含む）。
def n_species() -> int:
    return len(_vocab()[0]) + 1


# 技埋め込みの語彙数（不明用の ID 0 を含む）。
def n_moves() -> int:
    return len(_vocab()[1]) + 1


N_HASHED = HASH_BUCKETS + 2


# (ベース種族名のハッシュ, 種族値) -> フォルム名 の対応表。
# 種族値だけからメガシンカ後のフォルムを逆算するために使う（下の effective_species 参照）。
@lru_cache(maxsize=1)
def _formes_by_stats() -> dict[tuple, str]:
    table: dict[tuple, str] = {}
    for sid, entry in GenData.from_gen(GEN).pokedex.items():
        base = zlib.crc32(entry.get("baseSpecies", entry["name"]).lower().encode())
        table.setdefault((base, tuple(sorted(entry["baseStats"].items()))), sid)
    return table


def effective_species(mon: Pokemon) -> str:
    """Species id including in-battle formes.

    poke-env keeps ``species`` at the base forme when an *opponent* Mega
    Evolves (only types/stats/ability are updated), so recover the forme from
    the base stats in that case.
    """
    dex = GenData.from_gen(GEN).pokedex
    entry = dex.get(mon.species)
    stats = mon.base_stats
    if entry is None or not stats or entry["baseStats"] == stats:
        return mon.species
    base = zlib.crc32(entry.get("baseSpecies", entry["name"]).lower().encode())
    return _formes_by_stats().get((base, tuple(sorted(stats.items()))), mon.species)


# 現在のフォルム（復元済み）がメガシンカかどうか。
def is_mega(mon: Pokemon) -> bool:
    entry = GenData.from_gen(GEN).pokedex.get(effective_species(mon), {})
    return entry.get("forme", "").startswith("Mega")


# ポケモンの種族 ID を引く。見つからなければベース種族名で引き直す。
def species_id(mon: Pokemon) -> int:
    species, _ = _vocab()
    sid = species.get(effective_species(mon))
    if sid is None:
        sid = species.get(mon.base_species, UNKNOWN_ID)
    return sid


# 技の ID を引く（不明なら 0）。
def move_id(move: Move) -> int:
    return _vocab()[1].get(move.id, UNKNOWN_ID)


# 道具・特性の名前を HASH_BUCKETS 個の ID にハッシュする
# （poke-env には道具・特性の ID 表が無いため）。0=不明、1=「持たない/無い」と判明済み。
def hashed_id(value: Optional[str]) -> int:
    if value is None or value == GenData.UNKNOWN_ITEM:
        return UNKNOWN_ID
    if value == "":
        return NONE_ID
    return 2 + zlib.crc32(value.encode()) % HASH_BUCKETS


# ---------------------------------------------------------------- layout

GLOBAL_DIM = (
    len(WEATHERS)
    + len(FIELDS)
    + 2 * len(SIDE_CONDITIONS)
    + 10  # turn, can_mega, used_mega, opp_used_mega, remaining x2, force_switch, trapped, wait, teampreview
)
POKEMON_NUM_DIM = (
    9  # present, own, active, fainted, hp, revealed, is_mega, level, selected_in_teampreview
    + len(STATUSES)
    + len(BOOST_KEYS)
    + len(TYPES)  # current types (updated on Mega Evolution)
    + len(STAT_KEYS)  # base stats (updated on Mega Evolution)
)
POKEMON_CAT_DIM = 3 + N_MOVES  # species, item, ability, move ids
MOVE_NUM_DIM = (
    11  # present, bp, acc, pp, priority, effectiveness, stab, heal, drain, recoil, available
    + len(CATEGORIES)
    + len(TYPES)
)

_O_GLOBAL = 0
_O_PNUM = _O_GLOBAL + GLOBAL_DIM
_O_PCAT = _O_PNUM + N_SLOTS * POKEMON_NUM_DIM
_O_MNUM = _O_PCAT + N_SLOTS * POKEMON_CAT_DIM
OBS_DIM = _O_MNUM + N_SLOTS * N_MOVES * MOVE_NUM_DIM

# Index of the "active" flag inside a pokemon's numeric block (used by the model
# to find which of our slots the move actions refer to).
PNUM_PRESENT = 0
PNUM_ACTIVE = 2


def split_obs(obs):
    """Split a (batch, OBS_DIM) array/tensor into its named blocks."""
    b = obs.shape[0]
    return {
        "global": obs[:, _O_GLOBAL:_O_PNUM],
        "pokemon_num": obs[:, _O_PNUM:_O_PCAT].reshape(b, N_SLOTS, POKEMON_NUM_DIM),
        "pokemon_cat": obs[:, _O_PCAT:_O_MNUM].reshape(b, N_SLOTS, POKEMON_CAT_DIM),
        "move_num": obs[:, _O_MNUM:].reshape(b, N_SLOTS, N_MOVES, MOVE_NUM_DIM),
    }


# ---------------------------------------------------------------- encoding


# `values` の固定リストに対して `item` を one-hot 化する。
def _one_hot(values: list, item) -> list[float]:
    return [1.0 if v == item else 0.0 for v in values]


# ポケモンのタイプ（最大2つ）を multi-hot 化する。
def _types_vec(types) -> list[float]:
    present = {t for t in types if t is not None}
    return [1.0 if t in present else 0.0 for t in TYPES]


# 場の状態（リフレクター、まきびし等）をエンコードする。
# 積み重なるもの（まきびし・どくびし）は段階数で正規化する。
def _side_vec(conditions: dict) -> list[float]:
    out = []
    for cond in SIDE_CONDITIONS:
        if cond not in conditions:
            out.append(0.0)
        elif cond in STACKABLE:
            out.append(conditions[cond] / STACKABLE[cond])
        else:
            out.append(1.0)
    return out


# 天候・フィールド・両陣営の場の状態・ターン数・メガシンカ使用状況・
# 残りポケモン数などの試合全体の状態をエンコードする。
def _encode_global(battle: AbstractBattle) -> list[float]:
    own_left = sum(not m.fainted for m in battle.team.values())
    # Unrevealed opponent Pokemon are still alive.
    opp_left = N_TEAM - sum(m.fainted for m in battle.opponent_team.values())
    return (
        [1.0 if w in battle.weather else 0.0 for w in WEATHERS]
        + [1.0 if f in battle.fields else 0.0 for f in FIELDS]
        + _side_vec(battle.side_conditions)
        + _side_vec(battle.opponent_side_conditions)
        + [
            min(battle.turn, 100) / 100.0,
            float(bool(battle.can_mega_evolve)),
            float(bool(battle.used_mega_evolve)),
            float(bool(battle.opponent_used_mega_evolve)),
            own_left / N_TEAM,
            opp_left / N_TEAM,
            float(bool(getattr(battle, "force_switch", False))),
            float(bool(getattr(battle, "trapped", False))),
            float(bool(getattr(battle, "_wait", False))),
            float(bool(getattr(battle, "teampreview", False))),
        ]
    )


# 1体分のポケモンの数値特徴（存在/HP/状態異常フラグ、能力ランク、
# 現在のタイプ、種族値）をエンコードする。
def _encode_pokemon(mon: Pokemon, own: bool) -> list[float]:
    stats = mon.base_stats or {}
    return (
        [
            1.0,
            float(own),
            float(mon.active),
            float(mon.fainted),
            float(mon.current_hp_fraction),
            float(mon.revealed),
            float(is_mega(mon)),
            (mon.level or 100) / 100.0,
            float(mon.selected_in_teampreview),
        ]
        + _one_hot(STATUSES, mon.status)
        + [mon.boosts.get(k, 0) / 6.0 for k in BOOST_KEYS]
        + _types_vec(mon.types)
        + [stats.get(k, 0) / 255.0 for k in STAT_KEYS]
    )


def champions_max_pp(move: Move) -> int:
    """Max PP under Champions rules (data/mods/champions/scripts.ts):
    base PP capped at 20, then (pp / 5 + 1) * 4 unless the move has noPPBoosts."""
    base = min(move.entry.get("pp", 1), CHAMPIONS_PP_CAP)
    if move.entry.get("noPPBoosts"):
        return base
    return int((base / 5 + 1) * 4)


# 残り PP の割合。必要な場合は poke-env が数えている本家ルールの PP を
# Champions ルールの上限に換算する。
def pp_fraction(move: Move, champions: bool) -> float:
    if not move.max_pp:
        return 1.0
    if not champions:
        return move.current_pp / move.max_pp
    # poke-env counts PP down from its own (mainline) max, so convert the
    # number of uses to the Champions max.
    used = move.max_pp - move.current_pp
    cap = champions_max_pp(move)
    return max(cap - used, 0) / cap


# 1つの技の数値特徴（威力・命中・PP・優先度、相手へのタイプ相性、
# タイプ一致、回復/吸収/反動、現在使えるか）をエンコードする。
def _encode_move(
    move: Move,
    user: Pokemon,
    target: Optional[Pokemon],
    available_ids: Optional[set[str]],
    champions: bool,
) -> list[float]:
    try:
        effectiveness = target.damage_multiplier(move) if target is not None else 1.0
    except Exception:  # unknown type chart entries (e.g. "???" type)
        effectiveness = 1.0
    eff = float(np.log2(effectiveness)) / 2.0 if effectiveness > 0 else -1.5
    accuracy = 1.0 if move.accuracy is True else float(move.accuracy)
    pp = pp_fraction(move, champions)
    return (
        [
            1.0,
            min(move.base_power, 300) / 150.0,
            accuracy,
            float(pp),
            move.priority / 5.0,
            eff,
            float(move.type in user.types),
            float(move.heal),
            float(move.drain),
            float(move.recoil),
            float(available_ids is not None and move.id in available_ids),
        ]
        + _one_hot(CATEGORIES, move.category)
        + _one_hot(TYPES, move.type)
    )


# バトル全体を OBS_DIM 次元の観測ベクトルにエンコードする（レイアウトはモジュール冒頭の docstring 参照）。
def encode_battle(battle: AbstractBattle) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[_O_GLOBAL:_O_PNUM] = _encode_global(battle)

    own = list(battle.team.values())[:N_TEAM]
    opp = list(battle.opponent_team.values())[:N_TEAM]
    own_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon
    available = {m.id for m in battle.available_moves}
    champions = "champions" in (battle.format or battle.battle_tag)

    slots: list[tuple[int, Pokemon, bool]] = [(i, m, True) for i, m in enumerate(own)]
    slots += [(N_TEAM + i, m, False) for i, m in enumerate(opp)]
    for slot, mon, is_own in slots:
        p = _O_PNUM + slot * POKEMON_NUM_DIM
        obs[p : p + POKEMON_NUM_DIM] = _encode_pokemon(mon, is_own)

        moves = list(mon.moves.values())[:N_MOVES]
        c = _O_PCAT + slot * POKEMON_CAT_DIM
        cat = [species_id(mon), hashed_id(mon.item), hashed_id(mon.ability)]
        cat += [move_id(m) for m in moves] + [UNKNOWN_ID] * (N_MOVES - len(moves))
        obs[c : c + POKEMON_CAT_DIM] = cat

        target = opp_active if is_own else own_active
        avail = available if (is_own and mon is own_active) else None
        for j, move in enumerate(moves):
            m = _O_MNUM + (slot * N_MOVES + j) * MOVE_NUM_DIM
            obs[m : m + MOVE_NUM_DIM] = _encode_move(move, mon, target, avail, champions)
    return obs


def teampreview_action_mask(battle: AbstractBattle) -> list[int]:
    """Action mask for a Team Preview pick: only switch slots (actions 0-5)
    for own Pokemon not yet selected this preview are legal."""
    own = list(battle.team.values())[:N_TEAM]
    switches = [int(not mon.selected_in_teampreview) for mon in own]
    switches += [0] * (N_TEAM - len(switches))
    return switches + [0] * (N_ACTIONS - N_TEAM)
