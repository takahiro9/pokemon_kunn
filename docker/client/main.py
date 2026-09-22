"""Connects two poke-env RandomPlayer bots to a Pokemon Showdown server and
lets them battle each other, as a smoke test that the client can talk to the
server over websockets.

Server connection is controlled via env vars so the same image works whether
the Showdown server is a Docker Compose service (SHOWDOWN_HOST=showdown) or
running on the host (SHOWDOWN_HOST=localhost).
"""

import asyncio
import os

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import RandomPlayer

SHOWDOWN_HOST = os.environ.get("SHOWDOWN_HOST", "showdown")
SHOWDOWN_PORT = os.environ.get("SHOWDOWN_PORT", "8000")
# NOTE: if random battles finish in only a few seconds, requesting several
# n_battles in a row can trip the server's "you challenged less than 10
# seconds after your last challenge" misclick guard, silently dropping that
# challenge and hanging forever (poke-env has no retry for it). Keep this at
# 1 for a reliable smoke test, or add your own challenge-spacing/backoff if
# you need bulk self-play.
BATTLE_FORMAT = os.environ.get("BATTLE_FORMAT", "gen9championsrandombattle")
N_BATTLES = int(os.environ.get("N_BATTLES", "1"))

# Local (unregistered) Showdown servers accept any username that isn't
# already registered, so authentication_url only needs to point somewhere
# valid - it's never used to check a password for these guest accounts.
server_configuration = ServerConfiguration(
    f"ws://{SHOWDOWN_HOST}:{SHOWDOWN_PORT}/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


async def main() -> None:
    # rand=True avoids "username already taken" clashes when the container
    # is restarted while the server still remembers the previous session.
    player_1 = RandomPlayer(
        account_configuration=AccountConfiguration.generate("bot1", rand=True),
        server_configuration=server_configuration,
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
    )
    player_2 = RandomPlayer(
        account_configuration=AccountConfiguration.generate("bot2", rand=True),
        server_configuration=server_configuration,
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
    )

    await player_1.battle_against(player_2, n_battles=N_BATTLES)

    print(f"Finished {player_1.n_finished_battles} battle(s)")
    print(f"bot1 wins: {player_1.n_won_battles}")
    print(f"bot2 wins: {player_2.n_won_battles}")


if __name__ == "__main__":
    asyncio.run(main())
