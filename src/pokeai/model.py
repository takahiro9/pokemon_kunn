"""Actor-critic network (roadmap Phase 1 設計).

Pokemon tokens (embeddings + numeric features + pooled move tokens) are mixed
with a global-field token by a small Transformer encoder. Action logits are
produced "pointer style" so each action is scored from the token it refers to:

* switch i  (actions 0-5)   <- our Pokemon token i
* move j    (actions 6-9)   <- move token j of our active Pokemon + context
* gimmick g (actions 10-25) <- same move token + a per-gimmick head
  (poke-env's layout; in Champions only mega = actions 10-13 is ever legal)

Invalid actions are masked out before the softmax.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from pokeai import encoding as E

N_ACTIONS = 26  # SinglesEnv.get_action_space_size(9)
N_GIMMICKS = 4  # mega, z-move, dynamax, tera
MASK_VALUE = -1e9


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    species_dim: int = 64
    move_dim: int = 32
    hashed_dim: int = 16


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class ActorCritic(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.species_emb = nn.Embedding(E.n_species(), cfg.species_dim, padding_idx=0)
        self.move_emb = nn.Embedding(E.n_moves(), cfg.move_dim, padding_idx=0)
        self.item_emb = nn.Embedding(E.N_HASHED, cfg.hashed_dim, padding_idx=0)
        self.ability_emb = nn.Embedding(E.N_HASHED, cfg.hashed_dim, padding_idx=0)

        self.move_enc = _mlp(cfg.move_dim + E.MOVE_NUM_DIM, d, d)
        self.poke_enc = _mlp(
            cfg.species_dim + 2 * cfg.hashed_dim + E.POKEMON_NUM_DIM + d, d, d
        )
        self.global_enc = _mlp(E.GLOBAL_DIM, d, d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)

        self.switch_head = _mlp(2 * d, d, 1)
        self.move_head = _mlp(3 * d, d, 1)
        self.gimmick_head = _mlp(3 * d, d, N_GIMMICKS)
        self.value_head = _mlp(d, d, 1)

    def _encode(self, obs: torch.Tensor):
        parts = E.split_obs(obs)
        cat = parts["pokemon_cat"].long()
        pnum = parts["pokemon_num"]
        b = obs.shape[0]

        move_ids = cat[:, :, 3:]  # (B, 12, 4)
        move_tok = self.move_enc(
            torch.cat([self.move_emb(move_ids), parts["move_num"]], dim=-1)
        )  # (B, 12, 4, d)
        move_present = parts["move_num"][..., 0:1]
        pooled_moves = (move_tok * move_present).sum(2) / move_present.sum(2).clamp(min=1)

        poke_tok = self.poke_enc(
            torch.cat(
                [
                    self.species_emb(cat[:, :, 0]),
                    self.item_emb(cat[:, :, 1]),
                    self.ability_emb(cat[:, :, 2]),
                    pnum,
                    pooled_moves,
                ],
                dim=-1,
            )
        )  # (B, 12, d)
        glob_tok = self.global_enc(parts["global"]).unsqueeze(1)
        tokens = torch.cat([glob_tok, poke_tok], dim=1)  # (B, 13, d)

        empty = pnum[:, :, E.PNUM_PRESENT] < 0.5
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=obs.device), empty], 1)
        h = self.transformer(tokens, src_key_padding_mask=pad)
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        return h[:, 0], h[:, 1 : 1 + E.N_TEAM], move_tok, pnum

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """Return (masked logits (B, 26), value (B,))."""
        ctx, own_tok, move_tok, pnum = self._encode(obs)
        b, d = ctx.shape

        switch_logits = self.switch_head(
            torch.cat([own_tok, ctx.unsqueeze(1).expand(-1, E.N_TEAM, -1)], -1)
        ).squeeze(-1)  # (B, 6)

        # Move actions refer to our active Pokemon's moves.
        active = pnum[:, : E.N_TEAM, E.PNUM_ACTIVE]  # (B, 6)
        active_idx = active.argmax(1)
        has_active = (active.sum(1) > 0).float().view(b, 1, 1)
        idx = torch.arange(b, device=obs.device)
        act_moves = move_tok[idx, active_idx] * has_active  # (B, 4, d)
        act_poke = (own_tok[idx, active_idx] * has_active.view(b, 1)).unsqueeze(1)
        move_in = torch.cat(
            [act_moves, act_poke.expand(-1, E.N_MOVES, -1), ctx.unsqueeze(1).expand(-1, E.N_MOVES, -1)],
            -1,
        )
        move_logits = self.move_head(move_in).squeeze(-1)  # (B, 4)
        gimmick = self.gimmick_head(move_in)  # (B, 4 moves, 4 gimmicks)
        gimmick_logits = (move_logits.unsqueeze(-1) + gimmick).transpose(1, 2).reshape(b, -1)

        logits = torch.cat([switch_logits, move_logits, gimmick_logits], dim=1)
        logits = logits.masked_fill(mask < 0.5, MASK_VALUE)
        value = self.value_head(ctx).squeeze(-1)
        return logits, value

    def get_value(self, obs, mask):
        return self.forward(obs, mask)[1]

    def get_action_and_value(self, obs, mask, action=None, deterministic=False):
        logits, value = self.forward(obs, mask)
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def save_checkpoint(path, model: ActorCritic, **extra) -> None:
    torch.save(
        {"model_state": model.state_dict(), "model_config": asdict(model.cfg), **extra},
        path,
    )


def load_checkpoint(path, device="cpu") -> tuple[ActorCritic, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
