#!/usr/bin/env bash
# Build the static site into ./site (gitignored). Local only — nothing published.
#
#   bash website/build.sh            # uses website/docs.sif
#   STRICT=1 bash website/build.sh   # fail on warnings/broken links (link sweep)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${SIF:-$REPO/website/docs.sif}"
cd "$REPO"

if [[ ! -f "$SIF" ]]; then
    echo "Docs image not found: $SIF"
    echo "Build it once:  apptainer build website/docs.sif website/docs.def"
    exit 1
fi

ARGS=(build -f website/mkdocs.yml)
[[ "${STRICT:-0}" == "1" ]] && ARGS+=(--strict)

apptainer run --bind "$REPO":/work --pwd /work "$SIF" "${ARGS[@]}"
echo "Built -> $REPO/site  (open site/index.html)"
