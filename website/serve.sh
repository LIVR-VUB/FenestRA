#!/usr/bin/env bash
# Live-reload preview of the docs site, served LOCALLY only (http://127.0.0.1:8000).
# Runs inside the isolated docs container — nothing is published to the web.
#
#   bash website/serve.sh                 # uses website/docs.sif, port 8000
#   PORT=8080 bash website/serve.sh       # alternative port
#   SIF=other.sif bash website/serve.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${SIF:-$REPO/website/docs.sif}"
PORT="${PORT:-8000}"
cd "$REPO"

if [[ ! -f "$SIF" ]]; then
    echo "Docs image not found: $SIF"
    echo "Build it once:  apptainer build website/docs.sif website/docs.def"
    exit 1
fi

exec apptainer run --bind "$REPO":/work --pwd /work "$SIF" \
    serve -f website/mkdocs.yml -a "127.0.0.1:$PORT"
