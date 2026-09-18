#!/usr/bin/env bash
# FenestRA all-in-one launcher (Linux and macOS).
#
# Linux users who already have Apptainer do not need this: the native install
# with the Singularity engine is faster and has no VNC layer. This script is for
# anyone who wants the same one-command experience Windows users get.
set -euo pipefail

# A folder passed on the command line wins over everything else:
#     containers/run_fenestra.sh /path/to/scans [/path/to/models]
# Written as `if` rather than `[ ... ] && ...` because this script runs under `set -e`, where a
# trailing test that evaluates false would abort the launcher.
DATA_DIR="${FENESTRA_DATA:-$HOME/FenestRA/data}"
MODEL_DIR="${FENESTRA_MODELS:-$HOME/FenestRA/models}"
if [ -n "${1:-}" ]; then DATA_DIR="$1"; fi
if [ -n "${2:-}" ]; then MODEL_DIR="$2"; fi
IMAGE="${FENESTRA_IMAGE:-livrvub/fenestra:latest}"
PORT="${FENESTRA_PORT:-6080}"

mkdir -p "$DATA_DIR" "$MODEL_DIR"

if ! docker version >/dev/null 2>&1; then
    echo "Docker is not running, or this user cannot talk to the daemon." >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    cat >&2 <<MSG
The image "$IMAGE" is not on this machine, and it is not on Docker Hub.
Build it once, from the repository root:

    docker build -t $IMAGE -f containers/Dockerfile.allinone .
MSG
    exit 1
fi

# Ask Docker to actually satisfy a GPU request rather than inferring from `docker info`. A
# registered nvidia runtime does not prove the toolkit can fulfil the request, and Docker Desktop
# does not necessarily report one at all. This costs about a second and is never wrong.
gpu_flag=()
if docker run --rm --gpus all --entrypoint /bin/true "$IMAGE" >/dev/null 2>&1; then
    gpu_flag=(--gpus all)
else
    cat >&2 <<'MSG'
WARNING: no NVIDIA container runtime found.
  HAT and SwinIR upsampling and GPU Cellpose will not work; CLAHE (CPU) still does.
  On macOS this cannot be fixed — Docker Desktop has no NVIDIA passthrough.
MSG
fi

# Count what is actually in each folder, on the host, before the container starts. An empty file
# dialog inside the app is identical whether the folder is empty, the wrong one is mounted, or the
# filenames do not match -- and only the launcher knows which host path is really being used.
# `|| var=0` because of `set -euo pipefail`: a find that exits non-zero must not abort the launch.
scans=$(find "$DATA_DIR" -maxdepth 1 -type f \( -name '*.jpk' -o -name '*.jpk-qi-image' \) 2>/dev/null | wc -l) || scans=0
models=$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*.pth' 2>/dev/null | wc -l) || models=0

echo
echo "Open http://localhost:${PORT} once the log says the desktop is ready."
echo "  image:       $IMAGE"
echo "  scans:       $DATA_DIR  -> /data"
echo "  checkpoints: $MODEL_DIR -> /models"
echo

if [ "$scans" -gt 0 ]; then
    echo "Found $scans scan(s) in $DATA_DIR"
else
    echo "WARNING: no .jpk or .jpk-qi-image files in $DATA_DIR"
    echo "  The Load JPK dialog will be empty. Copy your scans there, or restart with"
    echo "  FENESTRA_DATA=/path/to/your/scans pointing at the folder that already holds them."
    echo "  The mount is live, so files copied in now appear without a restart."
fi

if [ "$models" -gt 0 ]; then
    echo "Found $models checkpoint(s) in $MODEL_DIR"
else
    echo "WARNING: no .pth checkpoint in $MODEL_DIR"
    echo "  HAT and SwinIR cannot run without one; CLAHE (CPU) still works."
    echo "  Use FENESTRA_MODELS=/path/to/models, or copy the checkpoint there."
fi
echo

# Published to loopback only. Binding 0.0.0.0 would expose an unauthenticated
# remote desktop of this machine to the whole network.
exec docker run --rm --name fenestra "${gpu_flag[@]}" \
    --shm-size=8g \
    -e VNC_PASSWORD="${VNC_PASSWORD:-}" \
    -e SCREEN="${SCREEN:-}" \
    -p "127.0.0.1:${PORT}:6080" \
    -v "$DATA_DIR:/data" \
    -v "$MODEL_DIR:/models" \
    "$IMAGE"
