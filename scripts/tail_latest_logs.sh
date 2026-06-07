#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COUNT="${1:-3}"
LINES="${2:-80}"
LOG_DIR="$ROOT/logs"

if ! [[ "$COUNT" =~ ^[0-9]+$ && "$LINES" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [num_logs] [num_lines]"
    echo "Example: $0 5 120"
    exit 2
fi

if [[ ! -d "$LOG_DIR" ]]; then
    echo "No logs directory found: $LOG_DIR"
    exit 0
fi

mapfile -t FILES < <(find "$LOG_DIR" -type f -name "*.log" -printf "%T@ %p\n" | sort -nr | head -n "$COUNT" | cut -d' ' -f2-)

if (( ${#FILES[@]} == 0 )); then
    echo "No .log files found in $LOG_DIR"
    exit 0
fi

for file in "${FILES[@]}"; do
    echo
    echo "===== $file (last $LINES lines) ====="
    tail -n "$LINES" "$file"
done
