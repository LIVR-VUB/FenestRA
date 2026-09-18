#!/usr/bin/env bash
# FenestRA all-in-one launcher (Linux and macOS).
#
# Linux users who already have Apptainer do not need this: the native install
# with the Singularity engine is faster and has no VNC layer. This script is for
# anyone who wants the same one-command experience Windows users get.
set -euo pipefail

DATA_DIR="${FENESTRA_DATA:-$HOME/FenestRA/data}"
MODEL_DIR="${FENESTRA_MODELS:-$HOME/FenestRA/models}"
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

gpu_flag=()
if docker info --format '{{.Runtimes}}' 2>/dev/null | grep -qi nvidia; then
    gpu_flag=(--gpus all)
else
    cat >&2 <<'MSG'
WARNING: no NVIDIA container runtime found.
  HAT and SwinIR upsampling and GPU Cellpose will not work; CLAHE (CPU) still does.
  On macOS this cannot be fixed — Docker Desktop has no NVIDIA passthrough.
MSG
fi

echo
echo "Open http://localhost:${PORT} once the log says the desktop is ready."
echo "  scans:       $DATA_DIR  -> /data"
echo "  checkpoints: $MODEL_DIR -> /models"
echo

# Published to loopback only. Binding 0.0.0.0 would expose an unauthenticated
# remote desktop of this machine to the whole network.
exec docker run --rm --name fenestra "${gpu_flag[@]}" \
    --shm-size=8g \
    -e VNC_PASSWORD="${VNC_PASSWORD:-}" \
    -p "127.0.0.1:${PORT}:6080" \
    -v "$DATA_DIR:/data" \
    -v "$MODEL_DIR:/models" \
    "$IMAGE"
