#!/usr/bin/env bash
# Bring the Solo Lite Rails API back (stopped to free memory for the KIT assistant).
# Run as root on bixops-prod-1:  bash /root/restore-bixops.sh
#
# Removing the KIT pieces (only if you want them gone; run as root):
#   systemctl disable --now kit-assistant
#   rm /etc/systemd/system/kit-assistant.service && systemctl daemon-reload
#   docker stop kit-pg && docker rm kit-pg
#   docker volume rm kit_pg_data              # deletes the KIT database for good
#   rm /home/deploy/kit-assistant.env /root/kit-pg.password
#   rm -rf /home/deploy/kit-voice-assistant
#   Caddy: delete the "kit.bixops.in { ... }" block from /etc/caddy/Caddyfile
#   (or copy back /etc/caddy/Caddyfile.bak-before-kit-*), then
#   caddy validate --config /etc/caddy/Caddyfile && systemctl reload caddy
set -euo pipefail

DB=solo-lite-be-db
CONTAINER=solo-lite-be-web-production-935517e4e82b25ed4d5e2b23d3334a4a9355e1f2

# Database first (a no-op if it is already running); wait until it accepts connections.
docker start "$DB"
ready=0
for i in $(seq 1 60); do
  if docker exec "$DB" pg_isready -q; then ready=1; break; fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  echo "$DB is not accepting connections after 120s; not starting the API." >&2
  docker logs --tail 20 "$DB" >&2
  exit 1
fi
echo "$DB accepting connections"

docker start "$CONTAINER"
for i in $(seq 1 30); do
  if curl -fsS -o /dev/null http://127.0.0.1:3000/up 2>/dev/null; then break; fi
  sleep 2
done
docker ps --filter "name=$CONTAINER" --format '{{.Names}} {{.Status}}'
curl -sI https://api.lite.bixops.in/up | head -n 1 || true
free -m
