#!/usr/bin/env bash
# Fetch third-party baseline implementations into vendor/.
#
# Idempotent: safe to re-run. Skips clones that already exist.

set -euo pipefail

VENDOR_DIR="$(cd "$(dirname "$0")" && pwd)/vendor"
mkdir -p "$VENDOR_DIR"

clone_if_missing() {
    local url="$1"
    local target="$2"
    if [ -d "$target" ]; then
        echo "[skip] $target already exists"
    else
        echo "[clone] $url -> $target"
        git clone --depth 1 "$url" "$target"
    fi
}

clone_if_missing https://github.com/clovaai/aasist "$VENDOR_DIR/aasist"

echo
echo "Done. Baseline code is in: $VENDOR_DIR"
echo "Refer to each project's README for any model-weight downloads."
