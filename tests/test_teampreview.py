"""Team Preview tests: mask/action-order handling and the model-driven pick
(reusing the switch head, no dedicated action head)."""

import logging

import numpy as np
import pytest
import torch
from poke_env.battle import Battle

from pokeai import encoding as E
from pokeai.env import PokemonEnv
from pokeai.model import ActorCritic
from pokeai.opponents import run_teampreview


def _mon(ident, details, condition="100/100"):
    return {
        "ident": ident, "details": details, "condition": condition, "active": False,
        "stats": {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        "moves": ["thunderbolt"], "baseAbility": "static", "item": "", "pokeball": "pokeball",
        "ability": "static", "commanding": False, "reviving": False,
        "teraType": "Normal", "terastallized": "",
    }


@pytest.fixture
def teampreview_battle() -> Battle:
    b = Battle("battle-gen9championsrandombattlelv50-1", "me", logging.getLogger("test"), gen=9)
    b.player_role = "p1"
    species = ["Pikachu", "Rotom-Wash", "Garchomp", "Gyarados", "Tyranitar", "Corviknight"]
    b.parse_request({
        "teamPreview": True,
        "side": {"name": "me", "id": "p1", "pokemon": [
            _mon(f"p1: {s.split('-')[0]}", f"{s}, L50") for s in species
        ]},
        "rqid": 1,
    })
    for s in ["Dragapult", "Landorus-Therian", "Clefable", "Heatran", "Toxapex", "Ferrothorn"]:
        b.parse_message(["", "poke", "p2", f"{s}, L50", ""])
    return b


def test_teampreview_action_mask_shrinks_as_picks_are_made(teampreview_battle):
    b = teampreview_battle
    mask = E.teampreview_action_mask(b)
    assert mask == [1, 1, 1, 1, 1, 1] + [0] * 20

    list(b.team.values())[2]._selected_in_teampreview = True
    mask = E.teampreview_action_mask(b)
    assert mask == [1, 1, 0, 1, 1, 1] + [0] * 20


def test_env_action_to_order_and_mask_use_teampreview_branch(teampreview_battle):
    b = teampreview_battle
    assert b.teampreview is True
    assert PokemonEnv.get_action_mask(b) == E.teampreview_action_mask(b)

    order = PokemonEnv.action_to_order(np.int64(3), b, fake=False, strict=True)
    assert order.order is list(b.team.values())[3]


def test_encode_battle_works_during_teampreview(teampreview_battle):
    obs = E.encode_battle(teampreview_battle)
    assert obs.shape == (E.OBS_DIM,) and np.isfinite(obs).all()
    blocks = {k: v[0] for k, v in E.split_obs(obs[None]).items()}
    assert blocks["global"][-1] == 1.0  # teampreview flag
    # opponent slots carry species-only info (no moves revealed)
    assert blocks["pokemon_num"][6:, E.PNUM_PRESENT].tolist() == [1] * 6
    assert (blocks["pokemon_cat"][6:, 3:] == E.UNKNOWN_ID).all()


def test_run_teampreview_picks_distinct_slots_and_marks_them(teampreview_battle):
    b = teampreview_battle
    model = ActorCritic()
    result = run_teampreview(model, b, device="cpu", deterministic=False)

    assert result.action.shape == (E.TEAMPREVIEW_PICK,)
    assert len(set(result.action.tolist())) == E.TEAMPREVIEW_PICK  # no repeats
    assert result.obs.shape == (E.TEAMPREVIEW_PICK, E.OBS_DIM)
    assert torch.isfinite(torch.as_tensor(result.logprob)).all()

    picked = [i for i, mon in enumerate(b.team.values()) if mon.selected_in_teampreview]
    assert sorted(picked) == sorted(result.action.tolist())

    order_str = result.order.removeprefix("/team ")
    assert len(order_str) == 6 and sorted(order_str) == list("123456")
    assert [int(c) - 1 for c in order_str[:3]] == result.action.tolist()
