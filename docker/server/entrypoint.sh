#!/bin/sh
set -e

VOLUME_DEFAULTS="/app/volume-defaults"

# config/ and databases/ are bind-mounted as volumes (see
# docker-compose.yml). On first run against an empty host directory, restore
# the defaults baked into the image (config-example.js, formats.ts,
# avatars/, chat-plugins/, databases/schemas, ...) without clobbering
# anything already there.
for dir in config databases; do
  if [ -d "$VOLUME_DEFAULTS/$dir" ]; then
    cp -rn "$VOLUME_DEFAULTS/$dir"/* "/app/$dir"/ 2>/dev/null || true
  fi
done

# Pokemon Showdown requires config/config.js to exist to boot.
if [ ! -f "/app/config/config.js" ]; then
  echo "No config/config.js found, creating one from config-example.js"
  cp "/app/config/config-example.js" "/app/config/config.js"
fi

exec node pokemon-showdown "$@"
