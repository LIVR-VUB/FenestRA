#!/bin/bash
# Bring up a minimal X desktop, publish it over noVNC, and run FenestRA on it.
#
# Started by tini as PID 1, so the background jobs below get reaped and SIGTERM is forwarded.
set -euo pipefail

export DISPLAY="${DISPLAY:-:0}"
SCREEN="${SCREEN:-1600x1000x24}"   # docker run -e SCREEN=1920x1080x24

echo "FenestRA starting. Screen ${SCREEN}."

Xvfb "$DISPLAY" -screen 0 "$SCREEN" -nolisten tcp &

# Poll for readiness rather than sleeping a guessed number of seconds: X usually answers on the
# first attempt, and a fixed sleep is either too short on a loaded machine or wasted time.
for _ in $(seq 1 100); do
    xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && break
    sleep 0.1
done
if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    echo "ERROR: Xvfb never came up on $DISPLAY" >&2
    exit 1
fi

# Without a window manager, QFileDialog and QMessageBox appear undecorated and cannot be moved,
# which makes the Load JPK button effectively unusable.
openbox --sm-disable &

if [ -n "${VNC_PASSWORD:-}" ]; then
    # Subshell: a bare `umask 077` here would persist for the rest of this script, so every file
    # napari later writes into /data — every CSV, TIFF and workbook — would come out 0600 and
    # unreadable to the user's own account on the host.
    ( umask 077; printf '%s\n' "$VNC_PASSWORD" > /tmp/.vncpass )
    # rm: makes x11vnc delete the file immediately after reading it.
    AUTH=(-passwdfile rm:/tmp/.vncpass)
else
    AUTH=(-nopw)
fi

# -forever  keep serving after a browser tab closes; x11vnc's default is to exit
# -shared   a second tab does not kick the first
# -localhost port 5900 stays inside the container's network namespace
x11vnc -display "$DISPLAY" -rfbport 5900 -localhost \
       -forever -shared -noxdamage -quiet "${AUTH[@]}" &

# This 0.0.0.0 is inside the container, and is required for `docker run -p` to reach it. It is
# NOT the security boundary. The boundary is the host-side publish, which the launcher scripts
# write as -p 127.0.0.1:6080:6080. Publishing 6080 on all host interfaces instead would put an
# unauthenticated desktop - with a file dialog onto every mounted folder - on the local network.
websockify --web=/usr/share/novnc 0.0.0.0:6080 127.0.0.1:5900 >/dev/null 2>&1 &

# Say out loud whether the GPU arrived. A silent fall back to CPU is this project's signature
# failure: everything still runs, just slowly and, for the DL methods, not at all.
/opt/venv-gui/bin/python - <<'PY' || true
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
else:
    print("=" * 72)
    print("WARNING: no GPU visible inside the container.")
    print("  Cellpose will run on the CPU: minutes per image instead of seconds.")
    print("  HAT and SwinIR upsampling will fail outright. CLAHE (CPU) still works.")
    print("  Windows: install a current NVIDIA driver and enable the WSL 2 backend")
    print("           in Docker Desktop, then relaunch with --gpus all.")
    print("  macOS:   Docker Desktop has no NVIDIA passthrough. There is no fix.")
    print("=" * 72)
PY

if [ -z "$(ls -A /models 2>/dev/null)" ]; then
    echo "NOTE: /models is empty, so HAT and SwinIR have no checkpoint to load."
    echo "      Mount one:  -v <your model folder>:/models"
fi

cat <<'BANNER'

  ------------------------------------------------------------------
   FenestRA is ready.  Open this in your browser:

       http://localhost:6080

   Your scans are at /data, your checkpoints at /models.
  ------------------------------------------------------------------

BANNER

# exec last: the container lives exactly as long as the application does.
exec /opt/venv-gui/bin/python /usr/local/bin/fenestra-app.py "$@"
