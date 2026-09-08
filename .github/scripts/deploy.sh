#!/usr/bin/env bash
set -euo pipefail

required=(DEPLOY_HOST DEPLOY_USER DEPLOY_PATH DEPLOY_SSH_KEY DEPLOY_HEALTH_URL IMAGE)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required deployment setting: ${variable}" >&2
    exit 1
  fi
done

DEPLOY_PORT="${DEPLOY_PORT:-22}"
mkdir -p ~/.ssh
chmod 700 ~/.ssh
printf '%s\n' "$DEPLOY_SSH_KEY" > ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
ssh-keyscan -p "$DEPLOY_PORT" -H "$DEPLOY_HOST" >> ~/.ssh/known_hosts

ssh_opts=(-i ~/.ssh/id_ed25519 -p "$DEPLOY_PORT" -o BatchMode=yes -o StrictHostKeyChecking=yes)
remote="${DEPLOY_USER}@${DEPLOY_HOST}"

ssh "${ssh_opts[@]}" "$remote" "mkdir -p '$DEPLOY_PATH'"
scp "${ssh_opts[@]}" deploy/docker-compose.prod.yml "$remote:$DEPLOY_PATH/docker-compose.prod.yml"

ssh "${ssh_opts[@]}" "$remote" \
  "DEPLOY_PATH='$DEPLOY_PATH' IMAGE='$IMAGE' bash -s" <<'REMOTE'
set -euo pipefail
compose_file="$DEPLOY_PATH/docker-compose.prod.yml"
previous_image="$(docker compose -f "$compose_file" ps -q kieu-bot-ui | xargs -r docker inspect --format '{{.Config.Image}}' 2>/dev/null || true)"
rollback_file="$DEPLOY_PATH/.kieu-bot-previous-image"
printf '%s' "$previous_image" > "$rollback_file"

rollback() {
  echo "Deployment failed; rolling back."
  if [[ -n "$previous_image" ]]; then
    KIEU_BOT_IMAGE="$previous_image" docker compose -f "$compose_file" up -d --remove-orphans
  fi
  rm -f "$rollback_file"
}
trap rollback ERR

KIEU_BOT_IMAGE="$IMAGE" docker compose -f "$compose_file" pull
KIEU_BOT_IMAGE="$IMAGE" docker compose -f "$compose_file" up -d --remove-orphans
container_id="$(docker compose -f "$compose_file" ps -q kieu-bot-ui)"
test -n "$container_id"

for attempt in $(seq 1 30); do
  status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}starting{{end}}' "$container_id")"
  if [[ "$status" == healthy ]]; then
    trap - ERR
    echo "Deployment healthy: $IMAGE"
    exit 0
  fi
  if [[ "$status" == unhealthy ]]; then
    exit 1
  fi
  sleep 2
done

echo "Timed out waiting for container health."
exit 1
REMOTE

if ! curl --fail --silent --show-error --retry 5 --retry-delay 2 "$DEPLOY_HEALTH_URL"; then
  echo "External health check failed; rolling back the remote deployment." >&2
  ssh "${ssh_opts[@]}" "$remote" "DEPLOY_PATH='$DEPLOY_PATH' bash -s" <<'ROLLBACK'
set -euo pipefail
compose_file="$DEPLOY_PATH/docker-compose.prod.yml"
rollback_file="$DEPLOY_PATH/.kieu-bot-previous-image"
previous_image="$(cat "$rollback_file" 2>/dev/null || true)"
if [[ -n "$previous_image" ]]; then
  KIEU_BOT_IMAGE="$previous_image" docker compose -f "$compose_file" up -d --remove-orphans
fi
rm -f "$rollback_file"
ROLLBACK
  exit 1
fi

ssh "${ssh_opts[@]}" "$remote" "rm -f '$DEPLOY_PATH/.kieu-bot-previous-image'"
