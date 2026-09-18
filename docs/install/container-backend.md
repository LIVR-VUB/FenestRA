# 3 - Container backend

The HAT and SwinIR models depend on `basicsr`. The container is built on
`nvcr.io/nvidia/pytorch:23.01-py3`, which supplies Python 3.8 and PyTorch 1.14, and the recipe
additionally pins NumPy below 1.24 and `opencv-headless` 4.8.0.74 for `basicsr`. That stack cannot
coexist in one process with a current napari install. So deep learning upsampling runs inside a
container, and the plugin talks to it by writing a temporary `.tif`, running one subprocess, and
reading the result back.

You only need this page if you plan to use the **HAT** or **SwinIR** methods. The **CLAHE (CPU)**
method never launches a container.

!!! tip "There is a shorter route on Windows and macOS"

    The [all-in-one container](all-in-one.md) ships napari, the plugin, Cellpose *and* a
    deep-learning backend in one image you open in a browser. It needs nothing on the host but
    Docker Desktop, and it replaces this page's build entirely.

    It is not the same stack. Its bundled backend runs **torch 2.1.2**, because the torch
    1.13/1.14 wheels carry no PTX fallback and will not start on any GPU newer than sm_86 — no
    RTX 40-series, no H100.

!!! info "This page builds the reference stack"

    `containers/dl_upsampling.def` is unchanged, and both recipes on this page still build torch
    1.14 on `nvcr.io/nvidia/pytorch:23.01-py3` — the stack the method was developed and validated
    against. **Numbers intended for publication should be produced with the container built on
    this page**, not with the all-in-one image.

## Get the recipes

The container definitions are in the repository, not in the pip package, so clone the repository
first:

```bash
git clone https://github.com/LIVR-VUB/FenestRA.git
cd FenestRA
```

That gives you `containers/dl_upsampling.def` (Apptainer) and `containers/Dockerfile` (Docker).

## Build the image

=== "Linux (Apptainer)"

    Install Apptainer through your distribution, then build from the repository root:

    ```bash
    sudo apptainer build dl_upsampling.sif containers/dl_upsampling.def
    ```

    This writes `dl_upsampling.sif` into the current directory. Keep it somewhere stable, because
    you will point the plugin at it by full path.

    In napari, set **Engine** to `Singularity` and use the `...` button next to
    **Singularity (.sif):** to select that file.

    !!! note "The path field no longer resets (0.3.0)"

        Before 0.3.0 the field arrived pre-filled with a developer's home directory, and toggling
        **Engine** back to `Singularity` overwrote whatever you had typed. Since 0.3.0 the field
        starts empty, and each engine keeps its own value across toggles
        (`_widget.py:140-149`, `:303-329`). Set `FENESTRA_SIF` to pre-fill it with your own path.

=== "Windows / macOS (Docker)"

    Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) first.

    **On Windows, set up GPU passthrough before you build.** Docker can only reach an NVIDIA GPU
    through the WSL 2 backend, and the plugin always passes `--gpus all`:

    1. In Docker Desktop, enable *Settings → General → "Use the WSL 2 based engine"*. The Hyper-V
       backend has no GPU support, and `--gpus all` fails on it outright.
    2. Install or update the NVIDIA driver **on Windows itself**. Do not install a driver or the
       CUDA toolkit inside WSL — the Windows driver is what publishes the GPU into the distro.
    3. If you launch napari from inside WSL rather than from Windows, enable that distro under
       *Settings → Resources → WSL Integration*.
    4. Check the passthrough before building anything, because the build is slow and this is not:

        ```bash
        docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
        ```

        It must print your GPU table. `could not select device driver "" with capabilities:
        [[gpu]]` means the passthrough is not set up, and **Run Upsampling** will fail on the
        same flag.

    Then build from the `containers/` directory. No `sudo` is needed on Windows or macOS:

    ```bash
    cd containers
    docker build -t livrvub/dl-upsampling:latest -f Dockerfile ..
    ```

    The trailing `..` is the build context, so run the command from inside `containers/`.

    Expect the first build to be long and large: the base image is
    `nvcr.io/nvidia/pytorch:23.01-py3`, several gigabytes before the recipe's own layers. Raise
    Docker Desktop's disk allocation under *Settings → Resources* if the build runs out of space.

    On Windows, the four bind mounts described at the bottom of this page are host paths: your
    temp directory, the output directory, the folder holding the `.pth`, and site-packages. The
    WSL 2 backend shares these for you. On Hyper-V each drive must be shared explicitly, and an
    unshared drive mounts as an empty directory rather than raising an error.

    In napari, set **Engine** to `Docker`. The field next to it changes to **Docker Tag:** and is
    filled with `livrvub/dl-upsampling:latest` for you, or with `FENESTRA_DOCKER_IMAGE` if you set
    it (`_widget.py:31`). There is no file to browse for.

    !!! note "Rebuild if your image predates the ENTRYPOINT fix"

        See [the section below](#the-docker-entrypoint-fix) if you built this image from an older
        checkout.

!!! note "`livrvub/dl-upsampling:latest` is a local tag"

    It is not published on Docker Hub. `docker pull` will not find it. The name is only the label
    the build command above attaches to the image on your own machine, and the plugin's default
    text matches that label so the two line up after you build.

## The third engine, and why it is not this one

Since 0.3.0 the **Engine** dropdown offers `Singularity`, `Docker` and `Local (bundled)`
(`_widget.py:37`). `Local (bundled)` runs the inference script as a plain subprocess under a second
Python interpreter on the same filesystem, with no bind mounts and no container, because a
container cannot launch a container. It exists for the [all-in-one image](all-in-one.md).

On a normal host there is no such interpreter, and choosing it raises before anything runs
(`pipeline.py:116-121`):

```text
The bundled deep-learning environment was not found at /opt/venv-dl/bin/python.
The Local engine exists only inside the FenestRA all-in-one container.
Set FENESTRA_DL_PYTHON, or choose the Singularity or Docker engine.
```

`FENESTRA_DL_PYTHON` points it at another interpreter if you have built the deep-learning
environment yourself. That environment is yours to keep correct; nothing checks its versions.

## The Docker ENTRYPOINT fix

`containers/Dockerfile` used to end with `ENTRYPOINT ["python"]`. The plugin appends its own
`python /opt/dl_project/scripts/inference.py ...` as the command, and Docker concatenates
ENTRYPOINT with that command, so the container ran
`python python /opt/dl_project/scripts/inference.py` and died with:

```text
can't open file '/opt/python': [Errno 2] No such file or directory
```

Python was being handed the literal string `python` as the name of the script to execute.

The `ENTRYPOINT` line has been removed from the Dockerfile, so an image built from the current
repository is correct and needs nothing from you. **If you see the error above, your image was
built from an older checkout** — `git pull` and `docker build` again.

The image now inherits the NVIDIA base image's own entrypoint, which execs whatever command it is
given, so the plugin's `python ...` arrives intact.

The Apptainer path was never affected. `singularity exec` bypasses the container's `%runscript`, so
the explicit `python` is required there. The two engines genuinely need different argv, which is
how the mismatch arose in the first place. History in [Known issues](../caveats/known-issues.md).

!!! note "`--gpus all` on macOS"

    The plugin passes `--gpus all` to `docker run`. Docker Desktop for macOS has no NVIDIA
    passthrough, so the flag has no hardware to expose. Deep learning upsampling on an Apple
    machine has no GPU available to it.

??? note "What gets mounted, for the curious"

    Both engines' argv is built in one place since 0.3.0 — `_build_dl_cmd()` at `pipeline.py:109`,
    called through `_run_dl_inference()` at `:168` by both the interactive worker and the batch
    loop. Before 0.3.0 the two paths carried byte-for-byte copies of the same block and could
    drift apart silently.

    The plugin binds four directories into the container (`pipeline.py:137-142`) and passes only
    paths that live inside them:

    | Host | Container |
    |---|---|
    | the installed `fenestra/backend/` directory | `/opt/dl_project/scripts` |
    | the directory holding the temporary input `.tif` | `/tmp_in` |
    | the temporary output directory | `/tmp_out` |
    | the directory holding your `.pth` file | `/tmp_model` |

    The inference script ships inside the pip package, so `fenestra/backend/` is inside
    site-packages. You do not need the research repository checked out for inference itself, only
    for the container recipes. More detail in [Containers](../architecture/containers.md).

Next: [4 - Model weights](model-weights.md).
