"""Observation encoder tests (roadmap Phase 1 step 2): missing info, unknown
moves, forme changes / Mega Evolution, Champions PP, and alignment with
SinglesEnv's action indices. Battles use the Champions Random Battle format."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from poke_env.battle import Battle, PokemonType
from poke_env.environment import SinglesEnv

from pokeai import encoding as E
from pokeai.model import N_ACTIONS, ActorCritic


def _mon(ident, details, condition, moves, active=False, item="leftovers", ability="static", tera="Normal"):
    return {
        "ident": ident, "details": details, "condition": condition, "active": active,
        "stats": {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        "moves": moves, "baseAbility": ability, "item": item, "pokeball": "pokeball",
        "ability": ability, "commanding": False, "reviving": False,
        "teraType": tera, "terastallized": "",
    }


def _move(mid, pp=16, disabled=False):
    return {"move": mid, "id": mid, "pp": pp, "maxpp": 16, "target": "normal", "disabled": disabled}


@pytest.fixture
def battle() -> Battle:
    b = Battle("battle-gen9championsrandombattle-1", "me", logging.getLogger("test"), gen=9)
    b.player_role = "p1"
    b.parse_request({
        "active": [{
            "moves": [_move("thunderbolt"), _move("voltswitch", disabled=True), _move("surf")],
            "canMegaEvo": True,
        }],
        "side": {"name": "me", "id": "p1", "pokemon": [
            _mon("p1: Pikachu", "Pikachu, L93, M", "200/250",
                 ["thunderbolt", "voltswitch", "surf"], active=True, item="lightball"),
            _mon("p1: Rotom", "Rotom-Wash, L86", "0 fnt", ["hydropump"], item="", ability="levitate"),
            _mon("p1: Garchomp", "Garchomp, L80, F", "300/300", ["earthquake", "dragonclaw"], ability="roughskin"),
        ]},
        "rqid": 3,
    })
    opp = b.get_pokemon("p2: Gyarados", details="Gyarados, L84, F")
    opp.switch_in(details="Gyarados, L84, F")
    return b


def _blocks(battle):
    obs = E.encode_battle(battle)
    assert obs.shape == (E.OBS_DIM,) and obs.dtype == np.float32
    assert np.isfinite(obs).all()
    return {k: v[0] for k, v in E.split_obs(obs[None]).items()}


def test_own_team_layout_matches_switch_actions(battle):
    p = _blocks(battle)
    num, cat = p["pokemon_num"], p["pokemon_cat"]
    # slots 0..2 are our team in battle.team order; 3..5 are padding
    assert num[:3, E.PNUM_PRESENT].tolist() == [1, 1, 1]
    assert num[3:6, E.PNUM_PRESENT].tolist() == [0, 0, 0]
    assert num[:3, E.PNUM_ACTIVE].tolist() == [1, 0, 0]
    assert num[1, 3] == 1 and num[1, 4] == 0  # Rotom fainted, 0 hp
    assert num[0, 4] == pytest.approx(0.8)
    assert cat[1, 1] == E.NONE_ID  # Rotom known to hold no item
    # SinglesEnv switch action i <-> team slot i
    mask = SinglesEnv.get_action_mask(battle)
    assert mask[:6] == [0, 0, 1, 0, 0, 0]  # only Garchomp can come in


def test_move_slots_match_move_actions(battle):
    p = _blocks(battle)
    moves = list(battle.active_pokemon.moves.values())
    assert [m.id for m in moves] == ["thunderbolt", "voltswitch", "surf"]
    assert p["pokemon_cat"][0, 3:6].tolist() == [E.move_id(m) for m in moves]
    available = p["move_num"][0, :, 10].tolist()
    assert available == [1, 0, 1, 0]  # voltswitch disabled, slot 4 empty
    mask = SinglesEnv.get_action_mask(battle)
    assert mask[6:10] == [1, 0, 1, 0]
    assert mask[10:14] == [1, 0, 1, 0]  # mega + move
    assert sum(mask[14:]) == 0  # no z-move / dynamax / tera in Champions
    assert _blocks(battle)["global"][-9] == 1.0  # can_mega_evolve


def test_opponent_partial_information(battle):
    p = _blocks(battle)
    num, cat = p["pokemon_num"], p["pokemon_cat"]
    assert num[6, E.PNUM_PRESENT] == 1 and num[6, 1] == 0  # present, not ours
    assert num[7:, E.PNUM_PRESENT].sum() == 0  # unrevealed
    assert cat[6, 1] == E.UNKNOWN_ID  # item not revealed yet
    assert (cat[6, 3:] == E.UNKNOWN_ID).all()  # no moves revealed yet
    # opponent's remaining count assumes unrevealed mons are alive
    assert p["global"][-5] == pytest.approx(1.0)


def test_revealed_and_unknown_moves(battle):
    gyarados = battle.opponent_active_pokemon
    gyarados.moved("waterfall")
    gyarados.moved("notarealmove")  # must not crash the encoder
    p = _blocks(battle)
    assert p["pokemon_cat"][6, 3] == E.move_id(gyarados.moves["waterfall"])
    # Thunderbolt vs Water/Flying: 4x -> log2(4)/2 = 1.0
    assert p["move_num"][0, 0, 5] == pytest.approx(1.0)


def test_mega_evolution_updates_species_types_and_flags(battle):
    before = _blocks(battle)
    battle.parse_message(["", "-mega", "p2a: Gyarados", "Gyarados", "Gyaradosite"])
    battle.parse_message(["", "detailschange", "p2a: Gyarados", "Gyarados-Mega, L84, F"])
    after = _blocks(battle)
    mon = battle.opponent_active_pokemon
    assert mon.species == "gyarados"  # poke-env keeps the base forme for opponents
    assert E.effective_species(mon) == "gyaradosmega" and E.is_mega(mon)
    assert before["pokemon_cat"][6, 0] != after["pokemon_cat"][6, 0]  # species id
    assert before["pokemon_num"][6, 6] == 0 and after["pokemon_num"][6, 6] == 1  # is_mega
    types = slice(9 + len(E.STATUSES) + len(E.BOOST_KEYS), 9 + len(E.STATUSES) + len(E.BOOST_KEYS) + len(E.TYPES))
    dark = E.TYPES.index(PokemonType.DARK)
    assert before["pokemon_num"][6][types][dark] == 0 and after["pokemon_num"][6][types][dark] == 1
    assert after["global"][-7] == 1.0  # opponent_used_mega_evolve


def test_champions_pp(battle):
    tbolt = battle.active_pokemon.moves["thunderbolt"]
    assert E.champions_max_pp(tbolt) == 16  # pp 15 -> (15/5+1)*4
    assert E.pp_fraction(tbolt, champions=True) == 1.0
    tbolt.use()
    assert E.pp_fraction(tbolt, champions=True) == pytest.approx(15 / 16)
    surf = battle.active_pokemon.moves["surf"]
    assert E.champions_max_pp(surf) == 16
    gyarados = battle.opponent_active_pokemon
    gyarados.moved("tackle")  # pp 35 in the main dex -> capped to 20 -> 20
    assert E.champions_max_pp(gyarados.moves["tackle"]) == 20
    assert E.pp_fraction(gyarados.moves["tackle"], champions=True) == pytest.approx(19 / 20)


def test_unknown_species_maps_to_unknown_id():
    unknown = SimpleNamespace(species="zzznotapokemon", base_species="zzznotapokemon", base_stats={})
    assert E.species_id(unknown) == E.UNKNOWN_ID
    forme = SimpleNamespace(species="notaformegarchomp", base_species="garchomp", base_stats={})
    assert E.species_id(forme) != E.UNKNOWN_ID  # falls back to base species


def test_model_respects_action_mask(battle):
    obs = torch.as_tensor(E.encode_battle(battle)).unsqueeze(0).repeat(64, 1)
    mask = torch.as_tensor(SinglesEnv.get_action_mask(battle), dtype=torch.float32)
    mask = mask.unsqueeze(0).repeat(64, 1)
    model = ActorCritic()
    action, logprob, _, value = model.get_action_and_value(obs, mask)
    assert action.shape == (64,) and value.shape == (64,)
    assert mask.shape[1] == N_ACTIONS
    assert bool(mask[torch.arange(64), action].all())
    assert torch.isfinite(logprob).all()
