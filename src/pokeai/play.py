"""Deploy a trained checkpoint onto a real Pokémon Showdown server so it can
actually battle — a human in the browser, a specific opponent, or the ladder.

Unlike ``evaluate.py`` (scripted bot-vs-bot benchmarks), this drives a single
``PolicyPlayer`` connection that waits for or initiates real battles:

    # Wait in the local Docker server for a human to challenge it in the
    # browser (http://localhost:8000). Prints the bot's username to type
    # into the challenge box; pick the same battle_format there.
    uv run python -m pokeai.play accept runs/<run>/checkpoints/latest.pt

    # Challenge a specific user (e.g. a second local client you're playing
    # from) n times.
    uv run python -m pokeai.play challenge runs/<run>/checkpoints/latest.pt SomeUsername -n 3

    # Ladder n games.
    uv run python -m pokeai.play ladder runs/<run>/checkpoints/latest.pt -n 10

By default this connects to the local server via ``pokeai.server``
(``SHOWDOWN_HOST``/``SHOWDOWN_PORT``, same as training/evaluate.py) using the
trained custom format, since that format (Lv50 Flat Rules) only exists
there. Pass ``--server showdown`` to instead play on the public
play.pokemonshowdown.com with a real (registered) ``--username``/
``--password`` and a format that actually exists there, e.g.
``--format gen9randombattle``.
"""

from __future__ import annotations

import argparse
import asyncio

from poke_env import AccountConfiguration, ServerConfiguration, ShowdownServerConfiguration

from pokeai.encoding import DEFAULT_FORMAT
from pokeai.opponents import make_policy_player
from pokeai.server import server_configuration as local_server_configuration


def resolve_server(spec: str) -> ServerConfiguration:
    """``"local"`` (default, ``pokeai.server``), ``"showdown"`` (the public
    server) or an explicit ``host[:port]`` of another Showdown server."""
    if spec == "local":
        return local_server_configuration()
    if spec == "showdown":
        return ShowdownServerConfiguration
    host, _, port = spec.partition(":")
    return ServerConfiguration(
        f"ws://{host}:{port or '8000'}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


async def run(args: argparse.Namespace) -> None:
    account_configuration = (
        AccountConfiguration(args.username, args.password) if args.username else None
    )
    player = make_policy_player(
        args.checkpoint,
        args.format,
        deterministic=args.deterministic,
        device=args.device,
        max_concurrent_battles=1,
        server_configuration=resolve_server(args.server),
        name=args.name,
        account_configuration=account_configuration,
        save_replays=args.save_replays,
    )
    print(f"connecting as {player.username!r} to {player.ps_client.websocket_url} "
          f"format={args.format!r} mode={args.mode}", flush=True)

    if args.mode == "accept":
        print(f"waiting for {args.n} challenge(s)"
              f"{f' from {args.opponent}' if args.opponent else ''} ...", flush=True)
        await player.accept_challenges(args.opponent, args.n)
    elif args.mode == "challenge":
        print(f"sending {args.n} challenge(s) to {args.opponent} ...", flush=True)
        await player.send_challenges(args.opponent, args.n)
    else:
        print(f"laddering {args.n} game(s) ...", flush=True)
        await player.ladder(args.n)

    print(
        f"done: {player.n_won_battles}/{player.n_finished_battles} won "
        f"(win rate {player.win_rate:.3f})",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["accept", "challenge", "ladder"])
    parser.add_argument("checkpoint", help="checkpoint path")
    parser.add_argument("opponent", nargs="?", help="username (required for accept-from-one/challenge)")
    parser.add_argument("-n", "--n-battles", dest="n", type=int, default=1)
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    parser.add_argument("--server", default="local", help="local | showdown | host[:port]")
    parser.add_argument("--username", help="registered Showdown username (needed to ladder on the public server)")
    parser.add_argument("--password", help="password for --username")
    parser.add_argument("--name", help="prefix for an auto-generated guest username (ignored if --username is set)")
    parser.add_argument("--deterministic", action="store_true", help="argmax instead of sampling")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-replays", action="store_true")
    args = parser.parse_args()

    if args.mode == "challenge" and not args.opponent:
        parser.error("challenge requires an opponent username")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
