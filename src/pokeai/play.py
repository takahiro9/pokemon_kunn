"""学習済みチェックポイントを実際の Pokémon Showdown サーバに接続し、
実戦（ブラウザからの人間、特定の相手、ラダー）で戦わせる。

``evaluate.py``（スクリプト同士の自動対戦ベンチマーク）とは異なり、
1つの ``PolicyPlayer`` 接続で実際の対戦を待ち受け／開始する:

    # ローカルの Docker サーバで、ブラウザ（http://localhost:8000）からの
    # 人間の挑戦を待つ。ボットのユーザー名を表示するので、挑戦ボックスに
    # 入力し、同じ battle_format を選ぶ。
    uv run python -m pokeai.play accept runs/<run>/checkpoints/latest.pt

    # 特定のユーザー（例: 自分が操作するもう1つのローカルクライアント）に
    # n 回挑戦する。
    uv run python -m pokeai.play challenge runs/<run>/checkpoints/latest.pt SomeUsername -n 3

    # n 試合ラダーに潜る。
    uv run python -m pokeai.play ladder runs/<run>/checkpoints/latest.pt -n 10

既定では ``pokeai.server`` 経由でローカルサーバ（``SHOWDOWN_HOST``/
``SHOWDOWN_PORT``、学習・評価と同じ）に、学習用のカスタムフォーマットで接続する
（Lv50 Flat Rules フォーマットはそこにしか存在しないため）。代わりに公開の
play.pokemonshowdown.com で遊ぶ場合は ``--server showdown`` を指定し、
実在する（登録済みの）``--username``/``--password`` と、向こうに実在する
フォーマット（例: ``--format gen9randombattle``）を指定する。
"""

from __future__ import annotations

import argparse
import asyncio

from poke_env import AccountConfiguration, ServerConfiguration, ShowdownServerConfiguration

from pokeai.encoding import DEFAULT_FORMAT
from pokeai.opponents import make_policy_player
from pokeai.server import server_configuration as local_server_configuration


def resolve_server(spec: str) -> ServerConfiguration:
    """``"local"``（既定、``pokeai.server``）、``"showdown"``（公開サーバ）、
    または他の Showdown サーバの明示的な ``host[:port]``。"""
    if spec == "local":
        return local_server_configuration()
    if spec == "showdown":
        return ShowdownServerConfiguration
    host, _, port = spec.partition(":")
    return ServerConfiguration(
        f"ws://{host}:{port or '8000'}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


# チェックポイントをサーバに接続し、accept/challenge/ladder のいずれかで実際に対戦する。
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


# CLI エントリポイント: 引数をパースして対戦セッションを実行する。
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
