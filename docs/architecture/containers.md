# Containers

The super-resolution half of FenestRA runs in its own Python environment, because `basicsr` will not coexist with the NumPy and torch the GUI needs. This page describes the three ways FenestRA reaches that environment, what is in each image, how the host mounts your files into it, and why the commands are not interchangeable.

## Three topologies

The Engine dropdown in panel 2 offers three (`_widget.py:37`):

| Engine | The DL environment is | Built from | Contact surface |
|---|---|---|---|
| **Singularity** | a `.sif` image, launched per run | `containers/dl_upsampling.def` | `singularity exec --nv` plus four bind mounts |
| **Docker** | a local image tag, launched per run | `containers/Dockerfile` | `docker run --rm --gpus all` plus four `-v` mounts |
| **Local (bundled)** | a second virtual environment on the same filesystem | `containers/Dockerfile.allinone` | a plain subprocess, no mounts |

The first two are the reference stack and are unchanged. **Local (bundled)** is new in 0.3.0 and exists for the all-in-one image described below; outside that image it has nothing to run and says so.

In every case the boundary is the same: one `subprocess.run` and one temporary `.tif` on disk. The GUI never imports `basicsr`, torch 1.14 or the architecture code, and the DL environment never imports napari or Qt.

!!! note "One argv builder since 0.3.0"

    `_build_dl_cmd(engine, temp_in_path, temp_out_dir, container_path, model_path, architecture)` at `pipeline.py:109-165` returns `(argv, env)` for all three engines, and `_run_dl_inference()` at `pipeline.py:168-189` is what both the interactive `@thread_worker` and the synchronous batch loop call.

    Before 0.3.0 the container argv was written out twice, byte for byte, once for each path. Changing a mount or a flag in one copy and not the other made the batch results silently differ from the interactive ones. If you are reading older notes that cite `pipeline.py:68-99` and `pipeline.py:226-257`, those blocks no longer exist.

## What is in the reference image

`containers/dl_upsampling.def` is the Apptainer/Singularity version and `containers/Dockerfile` is the Docker port. Both build from the same base and install the same packages.

Base image: `nvcr.io/nvidia/pytorch:23.01-py3`, which is Python 3.8 with torch 1.14.

| Pin | Why |
|---|---|
| `numpy<1.24.0` | NumPy 1.24 removed the deprecated `np.float` alias. The older libraries in this stack still use it, and the recipe comment records this as the reason. It is force-reinstalled at the end of the build so nothing quietly upgrades it. |
| `opencv-python-headless==4.8.0.74` | The version the recipe pins as stable against basicsr. Headless because the container has no display, and `opencv-python` is uninstalled afterwards so the two cannot conflict. |
| `basicsr>=1.4.2` | The framework HAT and SwinIR are written against. It also supplies `basicsr.archs.swinir_arch.SwinIR`, which is the first import `inference.py` tries for SwinIR (`inference.py:57`). |
| `scipy<1.11.0`, `scikit-image<0.20.0`, `pandas<2.0.0`, `matplotlib<3.8.0`, `seaborn<0.13.0` | Held at versions that still work on Python 3.8 alongside the NumPy pin. |
| `realesrgan>=0.3.0`, `pyiqa`, `lpips`, `brisque`, `cellpose>=2.2` | Installed for the wider benchmarking work the image was originally built for. FenestRA does not call any of them. Segmentation runs on the host. |

HAT and SwinIR are not pip packages here. They are cloned from source into `/opt` during the build:

```bash
git clone https://github.com/JingyunLiang/SwinIR.git
git clone https://github.com/XPixelGroup/HAT.git
```

`PYTHONPATH` is then set to `/opt/SwinIR:/opt/HAT`.

??? note "How inference.py reaches the two architectures"
    For HAT, `inference.py:34-39` loads `/opt/HAT/hat/archs/hat_arch.py` directly with `importlib`, bypassing the package `__init__`. The source comment gives the reason: importing the package triggers a `rgb2ycbcr` error from basicsr. If that fails it falls back to `from hat.archs.hat_arch import HAT`.

    For SwinIR, it tries `from basicsr.archs.swinir_arch import SwinIR` first, then falls back to adding `/opt/SwinIR` to the path and importing `models.network_swinir` (`inference.py:56-63`).

    Weights are loaded with `load_state_dict(state_dict, strict=True)` (`inference.py:87`). A checkpoint whose architecture does not match raises there and the run stops. That is deliberate.

## The all-in-one image

`containers/Dockerfile.allinone`, new in 0.3.0, puts napari, Qt, the plugin, Cellpose and the super-resolution backend in **one** image and serves the desktop to a browser over noVNC on port 6080. It is the recommended route on Windows and macOS. Build and run instructions are on [The all-in-one container](../install/all-in-one.md); this section only covers what it does to the architecture.

The hub and spoke survives, but the spoke stops being a container. You cannot launch a container from inside a container, so the moment the GUI itself is containerised the nested `docker run` has nowhere to go — that is the whole reason the **Local (bundled)** engine exists. The two environments are instead two virtual environments in one filesystem:

| Path | Python | Holds | Used by |
|---|---|---|---|
| `/opt/venv-gui` | 3.10 | napari 0.7, PyQt6 6.11.0, torch 2.4/cu124, Cellpose 4.1.1, AFMReader 0.0.7 | napari, the widget, segmentation, quantification |
| `/opt/venv-dl` | 3.10 | torch 2.1.2, torchvision 0.16.2, basicsr 1.4.2, numpy<1.24, HAT and SwinIR cloned to `/opt` | `inference.py` only |

Separation is by process, not by container. `_build_dl_cmd` runs `/opt/venv-dl/bin/python` against the *same* `backend/inference.py` file the mounted engines use, and strips `PYTHONPATH` and `PYTHONHOME` from that subprocess's environment (`pipeline.py:134`) so the GUI venv's NumPy and torch cannot leak into the DL interpreter. The image places HAT and SwinIR on the DL interpreter's path with a `.pth` file in its own `site-packages` rather than with `PYTHONPATH`, precisely because `PYTHONPATH` is scrubbed.

!!! warning "The all-in-one image is not the reference stack"

    Its DL venv runs **torch 2.1.2 / torchvision 0.16.2**, not the reference stack's **torch 1.14** on `nvcr.io/nvidia/pytorch:23.01-py3`.

    The reason is hardware, not preference: the torch 1.13 and 1.14 wheels are compiled for sm_37 through sm_86 with no PTX fallback, so they do not start at all on an RTX 40-series card, an H100, or anything newer. Pinning the reference version would have made the image unusable on most GPUs bought after 2022.

    `containers/dl_upsampling.def` still builds the reference stack and is unchanged. The architectures, the weights and the arithmetic are the same in both, but the two stacks are not bit-for-bit equivalent. **Numbers intended for publication should come from the reference container.**

## Bind mounts

Four mounts, identical for the Singularity and Docker engines (`pipeline.py:137-142`):

| Host path | Container path | Contents |
|---|---|---|
| `<site-packages>/fenestra/backend` | `/opt/dl_project/scripts` | `inference.py`, resolved from your installed package |
| directory holding the temporary input TIFF | `/tmp_in` | `temp_in.tif`, plus the `out/` subdirectory that is separately mounted at `/tmp_out` |
| temporary output directory | `/tmp_out` | where the result is written |
| parent directory of your `.pth` file | `/tmp_model` | the model weights |

The container is given the *directory* containing your weights, not the file, and the file is then addressed as `/tmp_model/<basename>`. These four are the only paths FenestRA asks for. Singularity additionally applies its own default binds (your home directory, `/tmp`, and the working directory) unless it is run with `--contain`, which FenestRA does not pass. The container never sees the `.jpk-qi-image`, only the TIFF the host wrote for it.

**The Local engine has no mounts at all.** There is one filesystem, so the real host paths are passed through unchanged and nothing has to be translated. That also removes the Windows path question: under Docker the `-v` arguments carry `C:\Users\...` paths straight through, and under Local there are no `-v` arguments.

## The command, per engine

Singularity, built from `pipeline.py:145-157`:

```bash
singularity exec --nv \
  --bind <site-packages>/fenestra/backend:/opt/dl_project/scripts \
  --bind <temp-input-dir>:/tmp_in \
  --bind <temp-output-dir>:/tmp_out \
  --bind <weights-parent-dir>:/tmp_model \
  <your-container.sif> \
  python /opt/dl_project/scripts/inference.py \
    --input /tmp_in \
    --output /tmp_out \
    --model_path /tmp_model/<weights.pth> \
    --arch hat \
    --tile_size 256
```

Docker, built from `pipeline.py:145-161`:

```bash
docker run --rm --gpus all \
  -v <site-packages>/fenestra/backend:/opt/dl_project/scripts \
  -v <temp-input-dir>:/tmp_in \
  -v <temp-output-dir>:/tmp_out \
  -v <weights-parent-dir>:/tmp_model \
  <your-docker-tag> \
  python /opt/dl_project/scripts/inference.py \
    --input /tmp_in \
    --output /tmp_out \
    --model_path /tmp_model/<weights.pth> \
    --arch hat \
    --tile_size 256
```

Local (bundled), built from `pipeline.py:123-130`:

```bash
/opt/venv-dl/bin/python <site-packages>/fenestra/backend/inference.py \
  --input <temp-input-dir> \
  --output <temp-output-dir> \
  --model_path <your-weights.pth> \
  --arch hat \
  --tile_size 256
```

`--nv` and `--gpus all` are the respective GPU passthrough flags; the Local engine needs neither, since it is not crossing a container boundary. `--arch` is `hat` or `swinir`, chosen from the Method dropdown. `--tile_size 256` is the `DL_TILE_SIZE` constant at `pipeline.py:100` and is written into all three argv shapes.

The interpreter the Local engine runs is `DEFAULT_DL_PYTHON`, `/opt/venv-dl/bin/python` (`pipeline.py:98`), overridable with the `FENESTRA_DL_PYTHON` environment variable. If that interpreter does not exist, `_build_dl_cmd` raises before anything is launched (`pipeline.py:116-121`):

```text
The bundled deep-learning environment was not found at <path>. The Local engine
exists only inside the FenestRA all-in-one container. Set FENESTRA_DL_PYTHON, or
choose the Singularity or Docker engine.
```

That is the expected message when Local is selected on an ordinary install. The container field is disabled and greyed out under this engine, because there is no container path to give (`_widget.py:303-329`).

## Why the engines need different argv

The recipes declare their default command differently, and that difference decides who supplies the word `python`.

The Singularity definition file defines a `%runscript` at `dl_upsampling.def:71-72`:

```text
%runscript
    exec python "$@"
```

`singularity exec` does not use `%runscript`. It runs the command you give it, so the explicit `python` in the argv above is required. The Singularity path is correct as written.

The Dockerfile declares no `ENTRYPOINT` of its own. Docker concatenates `ENTRYPOINT` with the command you pass, so an entrypoint of `python` would make the image supply one `python` and the host a second. The image instead inherits the NVIDIA base image's entrypoint, which execs the command as given, and the host's explicit `python` arrives intact.

The Local engine sidesteps the question: it names the interpreter itself, as `argv[0]`, so no default command is involved.

!!! warning "Images built before this fix still fail"
    The Dockerfile used to end with `ENTRYPOINT ["python"]`. The resulting argv inside the container was `python python /opt/dl_project/scripts/inference.py ...`. Python treated the literal string `python` as the script path and exited with:

    ```text
    can't open file '/opt/python': [Errno 2] No such file or directory
    ```

    An image built from the current repository does not have this problem. Seeing that error means the image predates the fix and needs rebuilding. Tracked on [Known Issues](../caveats/known-issues.md).

    The all-in-one image cannot produce this error at all, because it launches no second container.

## The output file

`inference.py:221` names the output after the input file's stem with a `_SR4x.tif` suffix, and writes it into the output directory. Since the host always writes its temporary input as `temp_in.tif`, the file that appears there is `temp_in_SR4x.tif`. The host does not look for that name: it globs `*.tif*` in the output directory and takes the first match (`pipeline.py:186-189`). This is the same for all three engines — under Singularity and Docker the directory is the one mounted at `/tmp_out`, under Local it is the host directory itself.
