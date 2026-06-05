#!/usr/bin/env bash
# Backup SQLite databases and config with rotation.
# Usage: ./scripts/backup.sh [backup_dir]
# Keeps last 7 daily backups.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${1:-${PROJECT_DIR}/backups}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_NAME="polymarket_backup_${TIMESTAMP}"
STAGING_DIR="${BACKUP_DIR}/${BACKUP_NAME}"
KEEP_COUNT=7

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cleanup() {
    rm -rf "${STAGING_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

log "Starting backup to ${BACKUP_DIR}"
mkdir -p "${STAGING_DIR}"

# ── Back up SQLite databases using .backup for consistency ──
for db in data/trades.db data/sentiment_cache.db; do
    src="${PROJECT_DIR}/${db}"
    if [ -f "$src" ]; then
        dest="${STAGING_DIR}/$(basename "$db")"
        if command -v sqlite3 &>/dev/null; then
            sqlite3 "$src" ".backup '${dest}'"
            log "Backed up ${db} (sqlite3 .backup)"
        else
            cp "$src" "$dest"
            log "Backed up ${db} (file copy — sqlite3 not available)"
        fi
    else
        log "Skipping ${db} (not found)"
    fi
done

# ── Back up flat files ──
for f in data/learned_weights.json config/settings.yaml data/alerts.jsonl; do
    src="${PROJECT_DIR}/${f}"
    if [ -f "$src" ]; then
        mkdir -p "${STAGING_DIR}/$(dirname "$f")"
        cp "$src" "${STAGING_DIR}/$(basename "$f")"
        log "Backed up ${f}"
    else
        log "Skipping ${f} (not found)"
    fi
done

# ── Create compressed archive ──
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" -C "${BACKUP_DIR}" "${BACKUP_NAME}"
rm -rf "${STAGING_DIR}"
log "Created archive: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"

# ── Rotate: keep last N backups ──
backup_count=$(find "${BACKUP_DIR}" -maxdepth 1 -name 'polymarket_backup_*.tar.gz' -type f | wc -l)
if [ "$backup_count" -gt "$KEEP_COUNT" ]; then
    delete_count=$((backup_count - KEEP_COUNT))
    find "${BACKUP_DIR}" -maxdepth 1 -name 'polymarket_backup_*.tar.gz' -type f -printf '%T+ %p\n' \
        | sort | head -n "$delete_count" | awk '{print $2}' \
        | while read -r old; do
            rm -f "$old"
            log "Rotated out: $(basename "$old")"
        done
fi

log "Backup complete. $(find "${BACKUP_DIR}" -maxdepth 1 -name 'polymarket_backup_*.tar.gz' -type f | wc -l) backups retained."
exit 0
