"""学習・評価で共有する Showdown サーバへの接続設定。

ホスト/ポートは環境変数から取るので、同じコードがホスト側
（SHOWDOWN_HOST=localhost、既定値）と docker compose 内
（SHOWDOWN_HOST=showdown）の両方で動く。
"""

import os

from poke_env import AccountConfiguration, ServerConfiguration


# SHOWDOWN_HOST / SHOWDOWN_PORT 環境変数から Showdown サーバへの WebSocket 接続設定を作る。
def server_configuration() -> ServerConfiguration:
    host = os.environ.get("SHOWDOWN_HOST", "localhost")
    port = os.environ.get("SHOWDOWN_PORT", "8000")
    return ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


# 指定した接頭辞から、他と衝突しないゲストアカウント名を生成する。
def account(prefix: str) -> AccountConfiguration:
    # Showdown のユーザー名は 18 文字までなので接頭辞は8文字に切り詰め、
    # rand=True で接尾辞を付けて並列プロセス/再起動時の名前衝突を防ぐ。
    return AccountConfiguration.generate(prefix[:8], rand=True)
