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
REM
REM Or, if your scans live somewhere else already: DRAG THAT FOLDER ONTO THIS
REM FILE in Explorer. It will be used instead, and nothing needs moving.
REM
REM The container can ONLY see the folders mounted below. napari's file dialog
REM cannot reach anything else on this PC, by design. To work from a different
REM folder without moving your scans, set FENESTRA_DATA before launching:
REM
REM     set FENESTRA_DATA=D:\Microscopy\LSEC
REM     containers\run_fenestra.bat
REM ---------------------------------------------------------------------------

REM A folder dragged onto this file, or passed on the command line, wins over everything else:
REM     containers\run_fenestra.bat D:\Microscopy\LSEC
REM Dragging a folder onto run_fenestra.bat in Explorer does exactly the same thing. %~1 strips
REM the quotes Explorer adds around a path containing spaces.
if not "%~1"=="" set "FENESTRA_DATA=%~1"
if not "%~2"=="" set "FENESTRA_MODELS=%~2"

if not defined FENESTRA_DATA   set "FENESTRA_DATA=%USERPROFILE%\FenestRA\data"
if not defined FENESTRA_MODELS set "FENESTRA_MODELS=%USERPROFILE%\FenestRA\models"
if not defined FENESTRA_IMAGE  set "FENESTRA_IMAGE=livrvub/fenestra:latest"
if not defined FENESTRA_PORT   set "FENESTRA_PORT=6080"

set "DATA_DIR=%FENESTRA_DATA%"
set "MODEL_DIR=%FENESTRA_MODELS%"
set "IMAGE=%FENESTRA_IMAGE%"
set "PORT=%FENESTRA_PORT%"

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
REM Count the scans on the Windows side, before the container starts. An empty Load JPK dialog
REM inside the app is indistinguishable from a wrong folder, a wrong extension, or an empty one,
REM and the launcher is the only place that knows which Windows path is really being mounted.
set "SCANCOUNT=0"
for /f %%N in ('dir /b /a-d "%DATA_DIR%\*.jpk" "%DATA_DIR%\*.jpk-qi-image" 2^>nul ^| find /c /v ""') do set "SCANCOUNT=%%N"

if "%SCANCOUNT%"=="0" (
  echo.
  echo WARNING: no .jpk or .jpk-qi-image files found in
  echo     %DATA_DIR%
  echo.
  echo   The "Load JPK.qi-image" dialog inside FenestRA will be empty, because that
  echo   folder is the only place it can look. Two common reasons:
  echo.
  echo   1. Your scans are in a different folder. Point FenestRA at it instead of
  echo      moving them, from this same Command Prompt:
  echo          set FENESTRA_DATA=D:\path\to\your\scans
  echo          containers\run_fenestra.bat
  echo.
  echo   2. Windows hides file extensions by default, so a file shown as
  echo      scan.jpk-qi-image may really be scan.jpk-qi-image.txt. Turn on
  echo      View ^> File name extensions in Explorer and check the real name.
  echo.
  echo   The folder is mounted live, so you can also just copy files into it now
  echo   and they will appear without restarting.
  echo.
) else (
  echo Found %SCANCOUNT% scan^(s^) in %DATA_DIR%
)

REM Same check for the checkpoints. A missing .pth is worse than a missing scan: HAT and SwinIR
REM fail outright, and an unreadable path in the DL Model box makes Cellpose fall back to its own
REM default model and produce perfectly plausible masks from the wrong network.
set "MODELCOUNT=0"
for /f %%N in ('dir /b /a-d "%MODEL_DIR%\*.pth" 2^>nul ^| find /c /v ""') do set "MODELCOUNT=%%N"

if "%MODELCOUNT%"=="0" (
  echo.
  echo WARNING: no .pth checkpoint found in
  echo     %MODEL_DIR%
  echo.
  echo   HAT and SwinIR upsampling cannot run without one. CLAHE ^(CPU^) still works.
  echo   Put your checkpoint there, or point FenestRA at the folder holding it:
  echo       set FENESTRA_MODELS=D:\path\to\your\models
  echo       containers\run_fenestra.bat
  echo.
  echo   The trained weights are not public yet; they ship with the manuscript.
  echo.
) else (
  echo Found %MODELCOUNT% checkpoint^(s^) in %MODEL_DIR%
)

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
