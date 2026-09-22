# syntax=docker/dockerfile:1

# --- Build stage: clone Pokemon Showdown and compile TypeScript to dist/ ---
FROM node:22-alpine AS builder

# Which branch/tag/commit of smogon/pokemon-showdown to build
ARG PS_VERSION=master

RUN apk add --no-cache git

WORKDIR /app

RUN git clone --branch "${PS_VERSION}" --depth 1 https://github.com/smogon/pokemon-showdown.git . \
    && npm ci \
    && node build

# --- Runtime stage ---
FROM node:22-alpine

RUN apk add --no-cache tini

WORKDIR /app

COPY --from=builder /app /app
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    # Keep a pristine copy of config/ and databases/ (config-example.js,
    # formats.ts, avatars/, chat-plugins/, databases/schemas, ...) so the
    # entrypoint can repopulate them into empty bind-mounted volumes on
    # first start.
    && mkdir -p /app/volume-defaults \
    && cp -r /app/config /app/volume-defaults/config \
    && cp -r /app/databases /app/volume-defaults/databases

# Config, chat/modlog/ticket logs and the sqlite ladder/battle databases
# should persist across rebuilds, so they are mounted as volumes via
# docker-compose.
VOLUME ["/app/config", "/app/logs/chat", "/app/logs/modlog", "/app/logs/tickets", "/app/databases"]

EXPOSE 8000

ENTRYPOINT ["tini", "--", "entrypoint.sh"]
CMD ["start"]
