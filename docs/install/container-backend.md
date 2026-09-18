# 3 - Container backend

The HAT and SwinIR models depend on `basicsr`. The container is built on
`nvcr.io/nvidia/pytorch:23.01-py3`, which supplies Python 3.8 and PyTorch 1.14, and the recipe
additionally pins NumPy below 1.24 and `opencv-headless` 4.8.0.74 for `basicsr`. That stack cannot
coexist in one process with a current napari install. So deep learning upsampling runs inside a
container, and the plugin talks to it by writing a temporary `.tif`, running one subprocess, and
reading the result back.

You only need this page if you plan to use the **HAT** or **SwinIR** methods. The **CLAHE (CPU)**
method never launches a container.

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

    !!! warning "The Engine dropdown overwrites what you typed"

        Switching **Engine** back to `Singularity` resets the path field to a hardcoded developer
        path. Re-select your `.sif` after any toggle. See
        [Known issues](../caveats/known-issues.md).

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
    filled with `livrvub/dl-upsampling:latest` for you. There is no file to browse for.

    !!! note "Rebuild if your image predates the ENTRYPOINT fix"

        See [the section below](#the-docker-entrypoint-fix) if you built this image from an older
        checkout.

!!! note "`livrvub/dl-upsampling:latest` is a local tag"

    It is not published on Docker Hub. `docker pull` will not find it. The name is only the label
    the build command above attaches to the image on your own machine, and the plugin's default
    text matches that label so the two line up after you build.

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

    The plugin binds four directories into the container and passes only paths that live inside
    them:

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
