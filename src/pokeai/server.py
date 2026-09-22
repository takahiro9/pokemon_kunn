"""Showdown server connection settings shared by training and evaluation.

The host/port come from env vars so the same code works on the host
(SHOWDOWN_HOST=localhost, the default) and inside docker compose
(SHOWDOWN_HOST=showdown).
"""

import os

from poke_env import AccountConfiguration, ServerConfiguration


def server_configuration() -> ServerConfiguration:
    host = os.environ.get("SHOWDOWN_HOST", "localhost")
    port = os.environ.get("SHOWDOWN_PORT", "8000")
    return ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


def account(prefix: str) -> AccountConfiguration:
    # Showdown usernames are capped at 18 chars; rand=True appends a suffix so
    # concurrent processes / restarts never collide on a name.
    return AccountConfiguration.generate(prefix[:8], rand=True)
