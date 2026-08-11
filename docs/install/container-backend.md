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

    Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), then build from the
    `containers/` directory. No `sudo` is needed on Windows or macOS:

    ```bash
    cd containers
    docker build -t livrvub/dl-upsampling:latest -f Dockerfile ..
    ```

    The trailing `..` is the build context, so run the command from inside `containers/`.

    In napari, set **Engine** to `Docker`. The field next to it changes to **Docker Tag:** and is
    filled with `livrvub/dl-upsampling:latest` for you. There is no file to browse for.

    !!! warning "This path does not currently work"

        See the section below before you spend time on it.

!!! note "`livrvub/dl-upsampling:latest` is a local tag"

    It is not published on Docker Hub. `docker pull` will not find it. The name is only the label
    the build command above attaches to the image on your own machine, and the plugin's default
    text matches that label so the two line up after you build.

## The Docker engine path is currently broken

`containers/Dockerfile` ends with `ENTRYPOINT ["python"]`, and the plugin appends its own
`python /opt/dl_project/scripts/inference.py ...` as the command. Docker concatenates ENTRYPOINT
and CMD, so the container tries to run `python python /opt/dl_project/scripts/inference.py` and
dies with:

```text
can't open file '/opt/python': [Errno 2] No such file or directory
```

Python is being handed the literal string `python` as the name of the script to execute.

Either one of these fixes it, and only one is needed:

```text
1. Edit containers/Dockerfile: replace ENTRYPOINT ["python"] with ENTRYPOINT [] (or CMD ["python"]),
   then rebuild the image.
2. Edit src/fenestra/pipeline.py: remove the "python" element from the Docker argv list.
   It appears in two places, the interactive block and the batch block, and both must be changed.
```

The Apptainer path is unaffected. `singularity exec` bypasses the container's `%runscript`, so the
explicit `python` is required there. The two engines genuinely need different argv, which is how
the mismatch arose. Tracked in [Known issues](../caveats/known-issues.md).

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
