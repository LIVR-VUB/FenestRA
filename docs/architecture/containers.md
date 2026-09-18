# Containers

The container holds the super-resolution half of FenestRA. This page describes what is in the image, why each version is pinned, how the host mounts your files into it, and why the Singularity and Docker commands are not interchangeable.

## What is in the image

Both recipes build from the same base and install the same packages. `containers/dl_upsampling.def` is the Apptainer/Singularity version and `containers/Dockerfile` is the Docker port.

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

## Bind mounts

Four mounts, identical for both engines:

| Host path | Container path | Contents |
|---|---|---|
| `<site-packages>/fenestra/backend` | `/opt/dl_project/scripts` | `inference.py`, resolved from your installed package |
| directory holding the temporary input TIFF | `/tmp_in` | `temp_in.tif`, plus the `out/` subdirectory that is separately mounted at `/tmp_out` |
| temporary output directory | `/tmp_out` | where the result is written |
| parent directory of your `.pth` file | `/tmp_model` | the model weights |

The container is given the *directory* containing your weights, not the file, and the file is then addressed as `/tmp_model/<basename>`. These four are the only paths FenestRA asks for. Singularity additionally applies its own default binds (your home directory, `/tmp`, and the working directory) unless it is run with `--contain`, which FenestRA does not pass. The container never sees the `.jpk-qi-image`, only the TIFF the host wrote for it.

## The command, per engine

Singularity, built from `pipeline.py:69-82`:

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

Docker, built from `pipeline.py:84-97`:

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

`--nv` and `--gpus all` are the respective GPU passthrough flags. `--arch` is `hat` or `swinir`, chosen from the Method dropdown. `--tile_size 256` is hardcoded in both blocks.

## Why the two engines need different argv

The two recipes declare their default command differently, and that difference decides who supplies the word `python`.

The Singularity definition file defines a `%runscript` at `dl_upsampling.def:71-72`:

```text
%runscript
    exec python "$@"
```

`singularity exec` does not use `%runscript`. It runs the command you give it, so the explicit `python` in the argv above is required. The Singularity path is correct as written.

The Dockerfile declares no `ENTRYPOINT` of its own. Docker concatenates `ENTRYPOINT` with the command you pass, so an entrypoint of `python` would make the image supply one `python` and the host a second. The image instead inherits the NVIDIA base image's entrypoint, which execs the command as given, and the host's explicit `python` arrives intact.

!!! warning "Images built before this fix still fail"
    The Dockerfile used to end with `ENTRYPOINT ["python"]`. The resulting argv inside the container was `python python /opt/dl_project/scripts/inference.py ...`. Python treated the literal string `python` as the script path and exited with:

    ```text
    can't open file '/opt/python': [Errno 2] No such file or directory
    ```

    An image built from the current repository does not have this problem. Seeing that error means the image predates the fix and needs rebuilding. Tracked on [Known Issues](../caveats/known-issues.md).

## The output file

`inference.py:221` names the output after the input file's stem with a `_SR4x.tif` suffix, and writes it into `/tmp_out`. Since the host always writes its temporary input as `temp_in.tif`, the file that appears in the output mount is `temp_in_SR4x.tif`. The host does not look for that name: it globs `*.tif*` in the output directory and takes the first match (`pipeline.py:263-266`).
