# The all-in-one container

One image. One command. Nothing installed on Windows except Docker Desktop.

This is the recommended route on **Windows and macOS**, and it exists because the step-by-step
install asks a biologist to assemble seven things that can each fail independently: Anaconda, a Qt
binding, a CUDA build of PyTorch, Cellpose, Git, the plugin, and a *second* container for the
deep-learning backend. Every one of those has produced a real support ticket.

Inside this image, none of them run on Windows at all.

## What it removes

| Failure you would otherwise hit | Why it cannot happen here |
|---|---|
| `qtpy.QtBindingsNotFoundError: No Qt bindings could be found` | Qt runs on Linux inside the image, against a pinned PyQt6 6.11.0 |
| `ImportError: DLL load failed while importing QtWidgets` | No Windows DLLs are involved |
| `DLL load failed ...: application control policies have blocked this file` | The `.so` files live on the container's filesystem, which AppLocker and WDAC do not police |
| `ERROR: Cannot find command 'git'` | `AFMReader` is installed from PyPI, so Git is no longer needed |
| `can't open file '/opt/python'` | There is no second container to launch |
| CPU-only PyTorch installed by accident | The CUDA build is baked in and checked at build time |
| Cellpose downloading 1.15 GB on first use | `cpsam` ships inside the image |

## What you will do

1. [Install Docker Desktop](#step-1-install-docker-desktop)
2. [Check your GPU](#step-2-check-your-gpu)
3. [Get the repository](#step-3-get-the-repository)
4. [Build the image, once](#step-4-build-the-image-once)
5. [Put your files where the app can see them](#step-5-put-your-files-where-the-app-can-see-them)
6. [Start it, and open your browser](#step-6-start-it-and-open-your-browser)

Steps 1 to 4 happen once. After that, using FenestRA is step 6 alone.

## Before you start

- **Docker Desktop**, with the **WSL 2 backend** on Windows.
- **An NVIDIA GPU and a current driver** for the deep-learning methods. Docker Desktop exposes it
  through `--gpus all`, which the launcher passes for you.
- **About 17 GB of disk** for the image, plus room for your data. The built image measured
  16.3 GB on 18 September 2026; most of it is two independent CUDA PyTorch stacks that cannot
  share anything, plus the 1.15 GB Cellpose model.
- **Git**, for step 3 only — to fetch the recipe. It is not a dependency of the software itself.

## Step 1 — Install Docker Desktop

Download it from [docker.com](https://www.docker.com/products/docker-desktop/) and install it.

On **Windows**, accept the **WSL 2 backend** during setup; it is the default and it is what makes
GPU passthrough possible. Then start Docker Desktop and wait for the whale icon in the system tray
to stop animating. Nothing below works until the daemon is actually running.

Confirm it:

```bash
docker version
```

## Step 2 — Check your GPU

Install a current **NVIDIA driver** for your card. You do not install CUDA, and you do not install
PyTorch — both live inside the image.

!!! warning "Without a GPU it still runs, just differently"

    FenestRA starts either way and says which case you are in, at startup, in the launcher window.

    - **CLAHE (CPU)** upsampling works normally.
    - **Cellpose** falls back to the CPU: minutes per image instead of seconds.
    - **HAT and SwinIR** cannot run at all.

!!! danger "macOS has no GPU path"

    Docker Desktop for macOS has no NVIDIA passthrough, so `--gpus all` has no hardware to expose.
    This is a limitation of Docker on macOS, not of FenestRA, and there is no workaround. The CLAHE
    route works; the deep-learning route does not.

## Step 3 — Get the repository

```bash
git clone https://github.com/LIVR-VUB/FenestRA.git
cd FenestRA
```

## Step 4 — Build the image, once

The image is not on Docker Hub — `docker pull` will not find it. You build it yourself, from the
recipe you just cloned:

=== "Windows"

    ```bat
    docker build -t livrvub/fenestra:latest -f containers\Dockerfile.allinone .
    ```

=== "Linux / macOS"

    ```bash
    docker build -t livrvub/fenestra:latest -f containers/Dockerfile.allinone .
    ```

It downloads several gigabytes and takes a while. You do it once.

The build is deliberately noisy about its own assumptions: it fails rather than completes if the
Qt xcb plugin cannot load, if `basicsr` cannot be imported, if the pinned HAT checkout does not
contain the `HAT` class, or if the Cellpose weights arrive truncated. A green build is therefore
worth something.

!!! bug "Linux: if Docker is installed from snap"

    The Canonical `docker` snap is confined and **cannot read `/media` or `/mnt` at all**. Building
    from a repository on an external or secondary drive fails with an empty context:

    ```
    #1 transferring dockerfile: 2B done
    ERROR: failed to solve: failed to read dockerfile: open Dockerfile.allinone: no such file or directory
    ```

    The file is right there; the daemon simply cannot see it. Either grant the interface:

    ```bash
    sudo snap connect docker:removable-media
    ```

    or clone the repository somewhere under your home directory and build from there.

### Updating later

```bash
git pull
docker build -t livrvub/fenestra:latest -f containers/Dockerfile.allinone .
```

Unchanged layers are reused, so an update is far quicker than the first build.

## Step 5 — Put your files where the app can see them

The launcher creates two folders on your machine the first time it runs, and mounts them into the
container:

| On your machine | Inside FenestRA | Put here |
|---|---|---|
| `%USERPROFILE%\FenestRA\data` / `~/FenestRA/data` | `/data` | your `.jpk-qi-image` scans, and your results |
| `%USERPROFILE%\FenestRA\models` / `~/FenestRA/models` | `/models` | your `.pth` checkpoints |

!!! warning "Save your results under /data"

    Anything you save outside those two folders lives inside the container and **disappears when
    you close it**. When the Quantify or Batch step asks where to save, choose somewhere under
    `/data`.

You can point the launcher somewhere else by editing the `DATA_DIR` and `MODEL_DIR` lines at the
top of `run_fenestra.bat`, or by setting `FENESTRA_DATA` and `FENESTRA_MODELS` before running
`run_fenestra.sh`.

## Step 6 — Start it, and open your browser

=== "Windows"

    Double-click `containers\run_fenestra.bat`.

=== "Linux / macOS"

    ```bash
    containers/run_fenestra.sh
    ```

Leave the window it opens alone — that is the application's log. Before napari appears it tells you
two things that otherwise fail quietly much later: whether a GPU was found, and whether `/models`
is empty. Closing that window, or pressing ++ctrl+c++ in it, shuts FenestRA down.

Then open **<http://localhost:6080>** in any browser. napari appears with the FenestRA dock already
open:

![The FenestRA desktop, served from the container to a browser](../assets/ui/all-in-one-desktop.png)

That screenshot is the real thing, captured from the running image on 18 September 2026 at
2560x1440 — not a mock-up. Note that it shows the CPU-only case: no `.jpk` is loaded and no GPU was
attached.

Panel 2 is already pointed at the bundled backend: **Engine** reads `Local (bundled)` and
**DL Model** is pre-filled with `/models/best_model_ema.pth`. From here, everything works exactly
as it does in the desktop app — follow the [User Guide](../guide/index.md).

!!! danger "Do not publish the port to the network"

    The launchers publish the desktop as `-p 127.0.0.1:6080:6080`, which means *this machine only*.

    If you change that to `-p 6080:6080` — the form most Docker tutorials show — Docker binds
    **every** network interface. Anyone on the same office LAN, conference wifi or hotel network
    can then open `http://your-machine:6080` and get a fully interactive napari session, including
    a file dialog that can browse every folder you mounted. VNC carries no encryption, so a
    password does not fix it either. There is no error and no warning; it simply works, for them.

    Set `VNC_PASSWORD` if you want a second layer, but the loopback publish is the actual control:

    === "Windows"

        ```bat
        set VNC_PASSWORD=something-long
        containers\run_fenestra.bat
        ```

    === "Linux / macOS"

        ```bash
        VNC_PASSWORD=something-long containers/run_fenestra.sh
        ```

## Model weights are not included

The image ships the *architecture*, not the *checkpoints*. Put your `.pth` files in the mounted
`models` folder; the DL Model field is pre-filled with `/models/best_model_ema.pth`.

Until the manuscript is published, the trained weights are not distributed. See
[Model weights](model-weights.md).

## How it works, and the one thing to know about it

FenestRA needs two Python environments that cannot be merged. The GUI needs napari, Qt 6 and
Cellpose 4 on a modern NumPy. The super-resolution backend needs `basicsr`, which still imports
`torchvision.transforms.functional_tensor` — a module deleted in torchvision 0.17 — so it is
pinned to an older pair.

Until now that separation was enforced by running the DL step in a **second container**. That is
exactly what stops working once the plugin itself is containerised: you cannot launch a container
from inside a container.

So the separation moved inside the image. Two virtual environments, one filesystem:

```
/opt/venv-gui   napari 0.7 · PyQt6 6.11 · torch 2.4 / cu124 · Cellpose 4.1.1 · AFMReader
/opt/venv-dl    torch 2.1.2 · torchvision 0.16.2 · basicsr 1.4.2 · HAT · SwinIR
```

The plugin's new **Local (bundled)** engine runs

```
/opt/venv-dl/bin/python .../fenestra/backend/inference.py --input ... --output ...
```

as an ordinary subprocess. Same process boundary, same isolation, no nesting. `PYTHONPATH` and
`PYTHONHOME` are stripped from that subprocess so the GUI environment's NumPy and PyTorch cannot
leak into the deep-learning interpreter.

!!! warning "This is not the reference stack, and that matters for publication"

    The validated backend — the one `containers/dl_upsampling.def` still builds, and the one the
    method was developed against — is **torch 1.14** on `nvcr.io/nvidia/pytorch:23.01-py3`.

    This image runs **torch 2.1.2** instead, because the torch 1.13/1.14 wheels are compiled for
    sm_37 through sm_86 with no PTX fallback: they do not start at all on an RTX 40-series card,
    an H100, or anything newer. Pinning the reference version would have made the image unusable
    on most GPUs bought after 2022.

    The architectures, the weights and the arithmetic are identical, but the two stacks are not
    bit-for-bit equivalent. **Numbers intended for publication should be produced with the
    reference container**; this image is for getting a working screen in front of a biologist.

## Resolution

**The desktop resizes itself to your browser window.** On connect, and again whenever you resize
the window, the browser asks the X server for a framebuffer of exactly that size, and napari's
window follows it. Nothing is stretched, so nothing is soft.

To get the most pixels: **maximise the browser window, then press ++f11++ for full screen.** The
desktop follows immediately. On a HiDPI screen, browser zoom works too — ++ctrl+minus++ gives you
more desktop at smaller text.

`SCREEN` sets only the size the desktop starts at, before a browser has connected:

```bash
docker run ... -e SCREEN=2560x1440 livrvub/fenestra:latest
```

Both `1920x1080` and the older `1920x1080x24` form are accepted.

!!! info "This changed after the first release of the image"

    The desktop used to be a fixed **1600x1000** framebuffer that noVNC scaled into the browser, so
    any browser window larger than that showed an upscaled, soft picture, and `SCREEN` was the only
    remedy. The display server is now TigerVNC's `Xvnc` rather than `Xvfb` + `x11vnc`, because
    `Xvfb`'s framebuffer cannot be resized after it starts and `Xvnc` implements the RFB
    `SetDesktopSize` extension. If your desktop is still a fixed size and scales, you are running an
    image built before that change: rebuild it.

## When something is wrong

Read the terminal the launcher opened. It states, before napari starts, whether a GPU was found
and whether `/models` is empty — the two things that otherwise fail quietly much later.

More symptoms, indexed by error message, on the
[Troubleshooting](../caveats/troubleshooting.md) page.
