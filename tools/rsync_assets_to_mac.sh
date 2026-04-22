#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/yolodist"
REMOTE="lizhechun@10.63.193.147"
PORT="22"
KEY="$HOME/.ssh/id_ed25519"
DEST="/Users/lizhechun/Documents/Codex/2026-04-20-hostname-pwd-whoami"
LOG="$ROOT/runs/overnight_logs/rsync_assets_to_mac.log"

mkdir -p "$ROOT/runs/overnight_logs"
cd "$ROOT"

log() {
  echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"
}

sync_once() {
  rsync -az --delete \
    -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8" \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='external/' \
    --exclude='__pycache__/' \
    runs/ "$REMOTE:$DEST/runs/" && \
  rsync -az \
    -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8" \
    weights/ "$REMOTE:$DEST/weights/"
}

for attempt in $(seq 1 120); do
  if timeout 12s ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -i "$KEY" -p "$PORT" "$REMOTE" "mkdir -p '$DEST/runs' '$DEST/weights' && echo ok" >/dev/null 2>&1; then
    log "SSH reachable, start rsync"
    if sync_once >> "$LOG" 2>&1; then
      log "rsync completed successfully"
      exit 0
    fi
    log "rsync failed after SSH connect, retry later"
  else
    log "SSH not reachable, retry later"
  fi
  sleep 300
done

log "rsync retries exhausted"
exit 1
