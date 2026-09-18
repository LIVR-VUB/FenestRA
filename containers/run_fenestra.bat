@echo off
setlocal

REM ---------------------------------------------------------------------------
REM FenestRA all-in-one launcher (Windows).
REM
REM Nothing is installed on Windows except Docker Desktop. napari, Qt, PyTorch,
REM Cellpose and the deep-learning backend all live inside the image, so none of
REM the usual Windows problems apply: no conda, no Qt DLLs, no Git, and no
REM application-control policy blocking unsigned .pyd files.
REM
REM Put your .jpk-qi-image scans in DATA_DIR and your .pth checkpoints in
REM MODEL_DIR, then double-click this file.
REM ---------------------------------------------------------------------------

set "DATA_DIR=%USERPROFILE%\FenestRA\data"
set "MODEL_DIR=%USERPROFILE%\FenestRA\models"
set "IMAGE=livrvub/fenestra:latest"
set "PORT=6080"

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

docker version >nul 2>&1
if errorlevel 1 (
  echo.
  echo Docker Desktop is not running. Start it, wait for the whale icon to stop
  echo animating, then run this file again.
  echo.
  pause
  exit /b 1
)

docker image inspect "%IMAGE%" >nul 2>&1
if errorlevel 1 goto :nomimage

REM Ask Docker to actually satisfy a GPU request, rather than inferring it from `docker info`.
REM A registered nvidia runtime does not prove the toolkit can fulfil the request, and Docker
REM Desktop does not necessarily report one at all. Parsing `docker info` is fragile too:
REM "{{.Runtimes}}" emits one ~10 KB line, past findstr's 8191-byte limit. This probe starts a
REM container that runs /bin/true and exits, so it answers the real question. Takes about a second.
echo Checking GPU access...
set "GPUFLAG=--gpus all"
docker run --rm --gpus all --entrypoint /bin/true "%IMAGE%" >nul 2>&1
if errorlevel 1 goto :nogpu
goto :launch

:nogpu
set "GPUFLAG="
echo.
echo WARNING: no NVIDIA container runtime was found.
echo   HAT and SwinIR upsampling and GPU Cellpose will NOT work.
echo   CLAHE (CPU) upsampling and CPU Cellpose still work.
echo   On Windows, fix this by installing a current NVIDIA driver and enabling
echo   the WSL 2 backend in Docker Desktop. On macOS there is no fix: Docker
echo   Desktop has no NVIDIA passthrough.
echo.

:launch
REM Only pass the variable through when it is actually set. cmd.exe leaves an undefined
REM %VAR% as the literal text "%VAR%", so the naive form would hand the container a password
REM of "%VNC_PASSWORD%" and lock the user out of their own desktop.
set "VNCFLAG="
if defined VNC_PASSWORD set "VNCFLAG=-e VNC_PASSWORD=%VNC_PASSWORD%"
REM SCREEN only sets the size the desktop starts at; the browser resizes it on connect.
set "SCREENFLAG="
if defined SCREEN set "SCREENFLAG=-e SCREEN=%SCREEN%"

echo.
echo Starting FenestRA. When the log below says the desktop is ready, open:
echo.
echo     http://localhost:%PORT%
echo.
echo Your scans:       %DATA_DIR%   (inside the app: /data)
echo Your checkpoints: %MODEL_DIR%  (inside the app: /models)
echo.
echo Close this window or press Ctrl+C to shut FenestRA down.
echo.

REM The port is published to 127.0.0.1 only. Do not change this to 0.0.0.0 or
REM to a bare -p 6080:6080: that would expose an unauthenticated remote desktop
REM of this machine to every other machine on your network.
docker run --rm --name fenestra %GPUFLAG% ^
  --shm-size=8g ^
  %VNCFLAG% %SCREENFLAG% ^
  -p 127.0.0.1:%PORT%:6080 ^
  -v "%DATA_DIR%":/data ^
  -v "%MODEL_DIR%":/models ^
  "%IMAGE%"

pause
exit /b 0

:nomimage
echo.
echo The image "%IMAGE%" is not on this machine.
echo It is not on Docker Hub either  -  you build it once, from the repository:
echo.
echo     git clone https://github.com/LIVR-VUB/FenestRA.git
echo     cd FenestRA
echo     docker build -t %IMAGE% -f containers\Dockerfile.allinone .
echo.
echo The build downloads several gigabytes and takes a while. You only do it once.
echo.
pause
exit /b 1
