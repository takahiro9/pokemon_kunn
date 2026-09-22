"""Benchmark evaluation (roadmap 0.2 共通評価プロトコル).

Win rate with a 95% Wilson interval, mean battle length and per-move inference
time of a checkpoint against the fixed baselines:

    uv run python -m pokeai.evaluate runs/<run>/checkpoints/latest.pt -n 500

Round robin + Elo between several agents (checkpoints and/or baselines):

    uv run python -m pokeai.evaluate --round-robin a.pt b.pt heuristic -n 200

Results are printed and written as JSON next to the (first) checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import time
from pathlib import Path

from poke_env.player import Player

from pokeai.encoding import DEFAULT_FORMAT
from pokeai.opponents import BASELINES, PolicyPlayer, make_policy_player
from pokeai.server import account, server_configuration

DEFAULT_OPPONENTS = ["random", "max_power", "heuristic"]


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def build_player(spec: str, fmt: str, concurrency: int, deterministic: bool) -> Player:
    """``spec`` is a baseline name or a checkpoint path."""
    if spec in BASELINES:
        return BASELINES[spec](
            account_configuration=account(spec),
            battle_format=fmt,
            server_configuration=server_configuration(),
            max_concurrent_battles=concurrency,
        )
    return make_policy_player(
        spec, fmt, deterministic=deterministic, max_concurrent_battles=concurrency,
        server_configuration=server_configuration(), name=Path(spec).stem,
    )


async def play(p1: Player, p2: Player, n: int) -> dict:
    """p1 vs p2 for n battles; stats are from p1's point of view."""
    p1.reset_battles()
    p2.reset_battles()
    if isinstance(p1, PolicyPlayer):
        p1.inference_seconds, p1.n_decisions = 0.0, 0
    start = time.time()
    await p1.battle_against(p2, n_battles=n)
    battles = list(p1.battles.values())
    wins = sum(bool(b.won) for b in battles)
    lo, hi = wilson(wins, len(battles))
    result = {
        "n": len(battles),
        "wins": wins,
        "win_rate": wins / max(1, len(battles)),
        "ci95": [lo, hi],
        "mean_turns": sum(b.turn for b in battles) / max(1, len(battles)),
        "wall_seconds": time.time() - start,
    }
    if isinstance(p1, PolicyPlayer) and p1.n_decisions:
        result["ms_per_move"] = 1000 * p1.inference_seconds / p1.n_decisions
    return result


def fit_elo(names: list[str], results: dict, iters: int = 500, anchor: str | None = None) -> dict:
    """Bradley-Terry fit (MM updates) of pairwise results, mapped to the Elo scale.

    ``results[(a, b)] = (wins_of_a, n)``. Ratings are centred on 1500, or
    ``anchor`` (e.g. ``random``) is pinned to 1000 when present.
    """
    wins = {x: 0.5 for x in names}  # +0.5 prior keeps unbeaten agents finite
    games: dict[tuple[str, str], int] = {}
    for (a, b), (w, n) in results.items():
        wins[a] += w
        wins[b] += n - w
        games[(a, b)] = games.get((a, b), 0) + n
        games[(b, a)] = games.get((b, a), 0) + n
    strength = {x: 1.0 for x in names}
    for _ in range(iters):
        new = {}
        for x in names:
            denom = sum(
                games.get((x, y), 0) / (strength[x] + strength[y]) for y in names if y != x
            )
            new[x] = wins[x] / denom if denom > 0 else strength[x]
        strength = new
    elo = {x: 400 * math.log10(s) for x, s in strength.items()}
    shift = 1000 - elo[anchor] if anchor in elo else 1500 - sum(elo.values()) / len(elo)
    return {x: round(v + shift, 1) for x, v in sorted(elo.items(), key=lambda kv: -kv[1])}


async def benchmark(args) -> dict:
    agent = build_player(args.checkpoint[0], args.format, args.concurrency, args.deterministic)
    out = {"checkpoint": args.checkpoint[0], "deterministic": args.deterministic, "results": {}}
    for opp_name in args.opponents:
        opp = build_player(opp_name, args.format, args.concurrency, args.deterministic)
        r = await play(agent, opp, args.n_battles)
        out["results"][opp_name] = r
        print(
            f"vs {opp_name:<10} win {r['win_rate']:.3f} "
            f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] n={r['n']} "
            f"turns={r['mean_turns']:.1f} ms/move={r.get('ms_per_move', float('nan')):.2f} "
            f"({r['wall_seconds']:.0f}s)",
            flush=True,
        )
    return out


async def round_robin(args) -> dict:
    specs = args.checkpoint + args.opponents
    names = [Path(s).stem if s not in BASELINES else s for s in specs]
    players = {n: build_player(s, args.format, args.concurrency, args.deterministic) for n, s in zip(names, specs)}
    results, table = {}, {}
    for a, b in itertools.combinations(names, 2):
        r = await play(players[a], players[b], args.n_battles)
        results[(a, b)] = (r["wins"], r["n"])
        table[f"{a} vs {b}"] = r
        print(f"{a} vs {b}: {r['wins']}/{r['n']} ({r['win_rate']:.3f})", flush=True)
    elo = fit_elo(names, results, anchor="random")
    print("Elo:", json.dumps(elo, indent=2))
    return {"pairs": table, "elo": elo}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="+", help="checkpoint path(s)")
    parser.add_argument("-n", "--n-battles", type=int, default=500)
    parser.add_argument("--opponents", nargs="*", default=None, help=f"baselines (default: {DEFAULT_OPPONENTS})")
    parser.add_argument("--round-robin", action="store_true")
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    parser.add_argument("--concurrency", type=int, default=8, help="simultaneous battles")
    parser.add_argument("--deterministic", action="store_true", help="argmax instead of sampling")
    parser.add_argument("--out", help="JSON output path")
    args = parser.parse_args()
    if args.opponents is None:
        args.opponents = DEFAULT_OPPONENTS

    out = asyncio.run(round_robin(args) if args.round_robin else benchmark(args))
    path = Path(args.out) if args.out else Path(args.checkpoint[0]).with_name(
        f"eval_{'rr' if args.round_robin else 'bench'}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
