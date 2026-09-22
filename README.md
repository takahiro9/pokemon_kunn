# pokemon_kunn — Pokémon battle AI

Implementation of the roadmap 「ポケモンバトルAI 実装ロードマップ（PPO→Rainbow→PPO×MCTS→LLM）」.
Currently covers **Phase 0 (environment + evaluation)** and **Phase 1 (PPO)**.

**Format:** Pokémon Champions singles, via a custom Showdown format
(`gen9championsrandombattlelv50`, defined in `data/config/custom-formats.ts`)
that layers the real Champions ranked/tournament **Flat Rules** — all Pokémon
forced to **Lv50**, Team Preview, Species Clause, Item Clause = 1, no
Mythical/Restricted Legendary, **bring 6 pick 3** — on top of the stock
`[Gen 9 Champions] Random Battle` random-team generator (sets from
[data/random-battles/champions](https://github.com/smogon/pokemon-showdown/tree/master/data/random-battles/champions)),
instead of that base format's unrestricted 6v6 at per-species levels 44-60.
Mega Evolution (no Terastal), PP capped at 20. Singles only. Legal actions are
switch ×6 (masked down to the 2 non-active Pokémon actually picked at Team
Preview, plus the active one), move ×4 and Mega + move ×4 (poke-env's
26-action layout with everything else masked). Team Preview (picking 3 of 6,
in lead order, from our own revealed sets and the opponent's revealed
species) is learned too: it reuses the same switch head/value head and is
folded into PPO as a Monte-Carlo-return transition once each episode
finishes (see `PokemonEnv._agent1_teampreview` in `src/pokeai/env.py`). The
opponent side still brings poke-env's default random 3 of 6.

## Layout

| Path | Contents |
|---|---|
| `docker-compose.yml`, `docker/server` | Local Pokémon Showdown server (`--no-security`: no auth, no throttling) |
| `docker/client` | poke-env RandomPlayer vs RandomPlayer smoke test |
| `src/pokeai/encoding.py` | Battle → observation vector (both teams, revealed info, field, per-move features, Mega state, Champions PP) |
| `src/pokeai/env.py` | `PokemonEnv` (poke-env `SinglesEnv` + our encoding/reward) and the per-episode opponent-mix wrapper |
| `src/pokeai/model.py` | Actor-critic: embeddings + Transformer over 12 Pokémon tokens, pointer-style masked action heads |
| `src/pokeai/opponents.py` | Baselines, `PolicyPlayer` (checkpoint → Player), self-play pool |
| `src/pokeai/ppo.py` | PPO trainer (CleanRL-style, parallel envs, curriculum → self-play) |
| `src/pokeai/evaluate.py` | Benchmarks (win rate + 95% Wilson CI, turns, ms/move) and round-robin Elo |
| `configs/` | `ppo_smoke.yaml` (pipeline check), `ppo_default.yaml` (Phase 1 run), `ppo_noshaping.yaml` (ablation) |

## Setup

```sh
docker compose up -d showdown   # Showdown on localhost:8000
uv sync
uv run pytest                   # encoder / masking unit tests (no server needed)
```

`SHOWDOWN_HOST` / `SHOWDOWN_PORT` override the server address (default `localhost:8000`).

## Train

```sh
uv run python -m pokeai.ppo --config configs/ppo_smoke.yaml     # ~1 min, checks the whole pipeline
uv run python -m pokeai.ppo --config configs/ppo_default.yaml   # Phase 1 run
uv run tensorboard --logdir runs
```

On a laptop, wrap long runs in `caffeinate -i` so macOS idle sleep doesn't pause training.

Each run writes `runs/<name>-<timestamp>/` with the resolved `config.yaml`,
TensorBoard logs (`win_rate/<opponent>` is the rolling training win rate),
`checkpoints/latest.pt` (+ periodic ones, resumable with `--resume`) and the
self-play `pool/`.

Opponent curriculum (`curriculum:` in the config) is a list of
`{until_step, mix}` stages. Mix keys: `random`, `max_power`, `heuristic`,
`latest` (newest self snapshot), `pool` (random past snapshot), `ckpt:<path>`.

## Evaluate

```sh
# vs RandomPlayer / MaxBasePowerPlayer / SimpleHeuristicsPlayer, 500 battles each
uv run python -m pokeai.evaluate runs/<run>/checkpoints/latest.pt -n 500
# round robin + Elo (random pinned to 1000)
uv run python -m pokeai.evaluate --round-robin runs/a/checkpoints/latest.pt runs/b/checkpoints/latest.pt -n 200
```

Results are also saved as JSON next to the checkpoint. Phase 1 target:
≥ 60% vs `heuristic` over 500 battles.

## Not done yet (Phase 1 backlog)

- LSTM/GRU variant for the history ablation (current model sees accumulated revealed info only)
- W&B logging (TensorBoard only)
- Throughput tuning: ~430 env steps/s with 16 envs on an M2; the Python side, not Showdown, is the bottleneck
- Opponent-side learned Team Preview: only our own agent's Team Preview pick is learned (see above); the opponent (baselines and self-play snapshots acting as `env.opponent`) still brings poke-env's default random 3 of 6, since `SingleAgentWrapper` only routes Team Preview through env actions for VGC. `PolicyPlayer.teampreview` (used for standalone evaluation/deployment) does use the learned pick.
