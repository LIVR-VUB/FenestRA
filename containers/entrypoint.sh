#!/bin/bash
# Bring up a minimal X desktop, publish it over noVNC, and run FenestRA on it.
#
# Started by tini as PID 1, so the background jobs below get reaped and SIGTERM is forwarded.
set -euo pipefail

export DISPLAY="${DISPLAY:-:0}"

# Initial size only. The browser resizes the desktop to its own window on connect, so this is
# what you see for the first instant and what a client that cannot resize is stuck with.
# Accepts Xvfb's old WxHxDEPTH form as well as plain WxH.
SCREEN="${SCREEN:-1920x1080x24}"
case "$SCREEN" in
    *x*x*) GEOMETRY="${SCREEN%x*}" ; DEPTH="${SCREEN##*x}" ;;
    *x*)   GEOMETRY="$SCREEN"      ; DEPTH=24 ;;
    *)     echo "ERROR: SCREEN must look like 1920x1080 or 1920x1080x24, got '$SCREEN'" >&2
           exit 1 ;;
esac

echo "FenestRA starting. Initial screen ${GEOMETRY}, depth ${DEPTH}."

if [ -n "${VNC_PASSWORD:-}" ]; then
    # Subshell: a bare `umask 077` would persist for the rest of this script, so every file napari
    # later writes into /data would come out 0600 and unreadable to the user on the host.
    # x11vnc's -storepasswd writes the standard VNC password format, which Xvnc reads; there is no
    # vncpasswd binary in the TigerVNC packages we install.
    ( umask 077; x11vnc -storepasswd "$VNC_PASSWORD" /tmp/.vncpass >/dev/null 2>&1 )
    AUTH=(-SecurityTypes VncAuth -PasswordFile /tmp/.vncpass)
else
    AUTH=(-SecurityTypes None)
fi

# Xvnc is the X server AND the VNC server in one process, replacing Xvfb + x11vnc. The reason is
# resolution: Xvfb has a fixed framebuffer, so the browser could only stretch it. Xvnc implements
# the RFB SetDesktopSize extension, so the desktop becomes exactly the size of the browser window
# and stays sharp. -localhost keeps port 5900 inside the container's network namespace.
Xvnc "$DISPLAY" -geometry "$GEOMETRY" -depth "$DEPTH" -rfbport 5900 -localhost \
     -AlwaysShared -desktop FenestRA "${AUTH[@]}" >/tmp/xvnc.log 2>&1 &

# Poll for readiness rather than sleeping a guessed number of seconds.
for _ in $(seq 1 100); do
    xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && break
    sleep 0.1
done
if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    echo "ERROR: Xvnc never came up on $DISPLAY" >&2
    cat /tmp/xvnc.log >&2
    exit 1
fi

# Without a window manager, QFileDialog and QMessageBox appear undecorated and cannot be moved,
# which makes the Load JPK button effectively unusable.
openbox --sm-disable &

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
