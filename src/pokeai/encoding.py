"""バトル状態 -> 固定長の観測ベクトルへの変換（ロードマップ 0.3 観測エンコーディング）。

対象フォーマット: [Gen 9 Champions] Random Battle（``gen9championsrandombattle``）、
すなわち Pokemon Champions のシングルバトル仕様（メガシンカありテラスタルなし、
技の PP は Champions 独自の計算式で最大 20 にキャップ）。

観測は gymnasium の Box にそのまま収まり安価にバッチ化できるよう、
flat な float32 ベクトルにしているが、``split_obs`` でモデル用のテンソルに
戻せるよう名前付きブロックとして並べている:

    [GLOBAL] [POKEMON_NUM x 12] [POKEMON_CAT x 12] [MOVE_NUM x 12 x 4]

ポケモンのスロット 0〜5 は ``battle.team`` の順（SinglesEnv の交代アクション
0〜5 と同じ順）の自分のチーム、6〜11 はこれまでに判明した相手のポケモン。
技のスロットは ``pokemon.moves`` の順（SinglesEnv の技アクション 6〜9 と同じ順）
に並べており、モデルがある行動がどの技・どの交代先を指すかを直接参照できる。

カテゴリ ID（種族・道具・特性・技）はベクトル内では float として保持され、
モデル側で埋め込みベクトルに変換される。ID 0 はパディング/不明を表す。
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
TEAMPREVIEW_PICK = 3  # Champions Flat Rules: 6匹選出して3匹使用
N_ACTIONS = 26  # SinglesEnv.get_action_space_size(9): 交代6 + 技4 x 5(通常+ギミック4種)

TYPES = list(PokemonType)
STATUSES = list(Status)
WEATHERS = list(Weather)
FIELDS = list(Field)
SIDE_CONDITIONS = list(SideCondition)
CATEGORIES = list(MoveCategory)
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]
STACKABLE = {SideCondition.SPIKES: 3, SideCondition.TOXIC_SPIKES: 2}

HASH_BUCKETS = 1024  # 道具・特性: poke-env に ID 表が無いためハッシュで割り当てる
UNKNOWN_ID = 0
NONE_ID = 1  # 「道具・特性を持たないと判明済み」を表す


# 種族名・技名 -> ID の対応表を作る（1 始まり。0 は不明/パディング用）。
@lru_cache(maxsize=1)
def _vocab() -> tuple[dict[str, int], dict[str, int]]:
    data = GenData.from_gen(GEN)
    # 0 はパディング/不明。実際のエントリは 1 から始まる。
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
    """戦闘中のフォルム変化を反映した種族名。

    *相手* がメガシンカした場合、poke-env は ``species`` をベースフォルムの
    ままにする（タイプ・種族値・特性だけが更新される）ため、その場合は
    種族値からフォルムを逆算して復元する。
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


# ---------------------------------------------------------------- レイアウト

GLOBAL_DIM = (
    len(WEATHERS)
    + len(FIELDS)
    + 2 * len(SIDE_CONDITIONS)
    + 10  # ターン数, メガシンカ可能/使用済み(自分), 相手使用済み, 残りポケモン数x2, 強制交代中, 交代不可, 行動待ち, チームプレビュー中
)
POKEMON_NUM_DIM = (
    9  # 存在, 自分か, 場にいるか, ひんしか, HP割合, 判明済みか, メガ済みか, レベル, チームプレビュー選出済みか
    + len(STATUSES)
    + len(BOOST_KEYS)
    + len(TYPES)  # 現在のタイプ（メガシンカで更新される）
    + len(STAT_KEYS)  # 種族値（メガシンカで更新される）
)
POKEMON_CAT_DIM = 3 + N_MOVES  # 種族, 道具, 特性, 技の各ID
MOVE_NUM_DIM = (
    11  # 存在, 威力, 命中, PP, 優先度, 相性, タイプ一致, 回復, 吸収, 反動, 使用可能か
    + len(CATEGORIES)
    + len(TYPES)
)

_O_GLOBAL = 0
_O_PNUM = _O_GLOBAL + GLOBAL_DIM
_O_PCAT = _O_PNUM + N_SLOTS * POKEMON_NUM_DIM
_O_MNUM = _O_PCAT + N_SLOTS * POKEMON_CAT_DIM
OBS_DIM = _O_MNUM + N_SLOTS * N_MOVES * MOVE_NUM_DIM

# ポケモンの数値ブロック内での「場にいるか」フラグの位置
# （技アクションがどの自分のスロットを指すかをモデルが判定するのに使う）。
PNUM_PRESENT = 0
PNUM_ACTIVE = 2


# (batch, OBS_DIM) の配列/テンソルを名前付きブロックに分割する。
def split_obs(obs):
    b = obs.shape[0]
    return {
        "global": obs[:, _O_GLOBAL:_O_PNUM],
        "pokemon_num": obs[:, _O_PNUM:_O_PCAT].reshape(b, N_SLOTS, POKEMON_NUM_DIM),
        "pokemon_cat": obs[:, _O_PCAT:_O_MNUM].reshape(b, N_SLOTS, POKEMON_CAT_DIM),
        "move_num": obs[:, _O_MNUM:].reshape(b, N_SLOTS, N_MOVES, MOVE_NUM_DIM),
    }


# ---------------------------------------------------------------- エンコーディング


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
    # まだ判明していない相手のポケモンは生きているものとして数える。
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
    """Champions ルールでの最大 PP（data/mods/champions/scripts.ts 準拠）:
    ベース PP を 20 でキャップした上で、noPPBoosts が無ければ (pp / 5 + 1) * 4。"""
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
    # poke-env は本家ルールの最大 PP を基準に数を減らしていくので、
    # 使用回数を Champions ルールの最大 PP に換算し直す。
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
    except Exception:  # タイプ相性表に無いタイプ（"???" 等）
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
    """チームプレビュー選出時のアクションマスク: まだ選んでいない自分のポケモン
    に対応する交代アクション（0〜5）のみを合法とする。"""
    own = list(battle.team.values())[:N_TEAM]
    switches = [int(not mon.selected_in_teampreview) for mon in own]
    switches += [0] * (N_TEAM - len(switches))
    return switches + [0] * (N_ACTIONS - N_TEAM)
