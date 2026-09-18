# Troubleshooting

Error messages you can hit while running FenestRA, with the cause and the fix for each. Sections are organized by the text you see, so you can search this page for the string in your dialog or terminal.

If the run completed but the numbers look wrong, this is the wrong page. Go to [Known issues](known-issues.md) or [The scale-domain question](scale-domain.md).

## The plugin does not appear under Plugins

There is no error text. The napari **Plugins** menu has no `FenestRA Pipeline` entry.

**Check the environment first.** napari has to be the one inside the environment where you installed the plugin:

```bash
conda activate fenestra-env
which napari
pip show napari-fenestra
python -c "import fenestra; print(fenestra.__file__)"
```

If `pip show` reports nothing, the plugin is installed somewhere else. Install it into the active environment.

**Restart napari.** napari discovers plugin manifests at startup. A plugin installed while napari was running does not appear until you close and reopen it.

**Check the manifest shipped.** The npe2 manifest is `napari.yaml` inside the package, and without it the widget is never registered:

```bash
python -c "import fenestra, os; print(os.path.exists(os.path.join(os.path.dirname(fenestra.__file__), 'napari.yaml')))"
```

If this prints `False`, reinstall:

```bash
pip install --force-reinstall napari-fenestra
```

**Look for the right name.** The manifest declares the plugin as `napari-fenestra` with the display name `FenestRA`, and contributes one widget whose display name is `FenestRA Pipeline`. See [Verify the installation](../install/verify.md).

## Could not load JPK

```text
Could not load JPK:
Failed to load JPK file: name 'load_jpk' is not defined
```

**Cause.** AFMReader is not installed in the active environment. `pipeline.py:15-18` catches the `ImportError` at import time and prints a warning to the terminal, so the plugin still loads and the failure only surfaces when you open a file:

```text
Warning: AFMReader not found or could not be imported.
```

**Fix.** AFMReader is installed from git, and is not pulled in by `pip install napari-fenestra`:

```bash
pip install git+https://github.com/AFM-SPM/AFMReader.git
```

See [Host environment](../install/host-environment.md).

**If AFMReader is installed** and the message after `Failed to load JPK file:` is something else, that text comes from AFMReader itself. The plugin always requests the `height_trace` channel (`pipeline.py:21-28`), so a file that does not carry that channel fails here.

## CUBLAS_STATUS_NOT_SUPPORTED

```text
CUBLAS_STATUS_NOT_SUPPORTED
```

This token appears inside a longer CUDA or cuBLAS error raised during the Cellpose step.

**Cause.** Your GPU has no BFloat16 hardware. Maxwell-generation cards, compute capability 5.2 and below, for example the Quadro M4000, are affected. Cellpose 4 defaults to `use_bfloat16=True` and FenestRA does not override it. That default was briefly disabled and then deliberately restored, to keep the acceleration on cards that do support it. The limitation is documented rather than worked around.

**Fix.** Run Cellpose on a GPU with BFloat16 support, or force it onto the CPU by hiding the GPU before launching napari:

```bash
CUDA_VISIBLE_DEVICES= napari
```

`pipeline.py:123` and `:281` choose the device from `torch.cuda.is_available()`, so this puts Cellpose on the CPU.

!!! warning
    With the Singularity engine the container subprocess inherits the same environment, so the deep-learning step also falls back to CPU and becomes very slow. If you need both, run the upsampling step in a session with the GPU visible and the segmentation step in a session without it.

    The Docker engine behaves differently: `pipeline.py:84-97` builds `docker run` with no `-e` flags, so no host variable is forwarded, and the image sets `ENV CUDA_VISIBLE_DEVICES=0` internally (`containers/Dockerfile:63`). Hiding the GPU from napari therefore does not hide it from a Docker-engine inference run.

## Container DL Inference failed

```text
Container DL Inference failed: <everything the container wrote to stderr>
```

Raised whenever the container process exits with a non-zero status, at `pipeline.py:110` for the interactive path and `pipeline.py:261` for the batch path. The dialog contains the container's full stderr, so scroll to the **last few lines**, which carry the actual Python traceback.

**Check these before reading the stderr:**

| Setting | What to confirm |
|---|---|
| DL Model | The `.pth` file exists on the host. If not, the plugin stops earlier with `Model path is invalid.` (`_widget.py:325`). |
| Singularity (.sif) | The file exists. If not, the plugin stops earlier with `Singularity container path is invalid.` (`_widget.py:331`). |
| Docker Tag | Not validated by the plugin. A typo reaches docker and comes back as a docker error. |
| Engine | Matches what is installed on your machine. |
| Method | HAT sends `--arch hat`, SwinIR sends `--arch swinir` (`_widget.py:322`). The checkpoint has to match the architecture. |

**Then read the stderr:**

| Last lines of stderr | Cause | Fix |
|---|---|---|
| `can't open file '/opt/python'` | The Docker image predates the ENTRYPOINT fix | Rebuild it. See the next section. |
| `Error(s) in loading state_dict`, with size mismatches | The checkpoint is a different architecture than the Method dropdown selected, or it uses `embed_dim=96` and `depths=[6]*4`, which the plugin cannot build (`inference.py:47-70`, `strict=True` at `:87`) | Switch the Method to match the checkpoint, or use an `embed_dim=180` checkpoint. See [Known issues](known-issues.md#7-only-embed_dim-180-checkpoints-load). |
| `ImportError: HAT arch could not be imported` | The container does not have HAT at `/opt/HAT` | Rebuild the container from the recipe in `containers/`. See [Container backend](../install/container-backend.md). |
| `SwinIR arch could not be imported` | The container has neither basicsr's SwinIR nor `/opt/SwinIR` | Rebuild the container. |
| `Padding size should be less than the corresponding input dimension` | SwinIR tiling on an unlucky image height | See the SwinIR section below. |

**About the `Background thread error:` prefix.** On the interactive **Run Upsampling** path the whole subprocess block is wrapped in a handler that re-raises everything with that prefix (`pipeline.py:101-114`), so *every* failure reaches you as `Background thread error: ...`, including an ordinary non-zero container exit. The prefix alone tells you nothing. What matters is the text after it:

```text
Background thread error: Container DL Inference failed: <container stderr>
```

means the container ran and the script inside it failed, so read the stderr and use the table above.

```text
Background thread error: [Errno 2] No such file or directory: 'docker'
```

means the container runtime itself could not be launched, for example selecting Docker on a machine that only has Apptainer.

The batch path raises the same message without the prefix (`pipeline.py:261`), under a dialog titled **Batch Error** rather than **DL Error**.

The interactive path wraps the subprocess call and reports it this way (`pipeline.py:113-114`). The batch path does not wrap it, so the same problem appears there as a bare `FileNotFoundError`.

## Docker cannot open the file /opt/python

```text
python: can't open file '/opt/python': [Errno 2] No such file or directory
```

**Cause.** Your Docker image was built from a checkout where `containers/Dockerfile` still ended with `ENTRYPOINT ["python"]`. The plugin appends its own `python /opt/dl_project/scripts/inference.py ...` as the command (`pipeline.py:91` and `:249`), and Docker concatenates ENTRYPOINT and CMD, so the container runs `python python /opt/dl_project/scripts/inference.py` and Python treats the literal word `python` as the script path.

**Fix.** The `ENTRYPOINT` line has been removed from the Dockerfile. Update the repository and rebuild:

```bash
git pull
cd containers
docker build -t livrvub/dl-upsampling:latest -f Dockerfile ..
```

Confirm the rebuilt image before returning to napari:

```bash
docker run --rm livrvub/dl-upsampling:latest python -c "print('ok')"
```

**On Linux**, you can also switch **Engine** to Singularity while you rebuild. That path was never affected, because `singularity exec` bypasses the container's runscript and needs the explicit `python`.

## No output generated from DL Upsampling

```text
No output generated from DL Upsampling.
```

Raised at `_widget.py:364` and `pipeline.py:265` when the container exited successfully but no `*.tif*` file appeared in the output directory.

**Cause 1: the input is a uniform image.** `inference.py:101-103` prints a warning and returns without writing when the scan has no height variation at all, and the script still exits 0:

```text
Warning: temp_in.tif is a uniform image. Skipping.
```

Check the raw layer in napari. A flat or failed scan produces this.

**Cause 2: the output bind mount did not resolve.** The plugin mounts the host temp directory into the container as `/tmp_out` (`pipeline.py:73`, `:88`). On Windows and macOS, Docker Desktop only shares directories you have allowed in its file-sharing settings, and the plugin uses the system temporary directory. Add that location to the shared paths.

**Cause 3, related trap.** The interactive path reuses one temporary directory for the whole napari session (`_widget.py:26`, `:361-362`) and loads the first `*.tif*` it finds. If a run writes nothing, an output from an earlier run in the same session can be loaded instead of this error being raised. If you are not certain which result you are looking at, restart napari and run once. The batch path is not affected, because it creates a fresh temporary directory per image (`pipeline.py:388`).

## Cellpose finds nothing, or finds everything

There is no error text. The `Cellpose Masks` layer is empty, or covers the whole field.

Adjust the two thresholds in panel 3:

| Symptom | Control | Direction |
|---|---|---|
| No masks at all | Cellprob Thresh | Lower it, into negative values. |
| No masks at all | Flow Thresh | Raise it, which accepts less regular shapes. |
| Small pores missing | Diameter | Set it below 30, which upscales the image before segmentation. |
| Far too many masks | Cellprob Thresh | Raise it. |
| Masks merged or ragged | Flow Thresh | Lower it. |

Three fixed behaviors also matter here:

- **Diameter is a rescale factor, not a size.** Cellpose 4 computes `image_scaling = 30 / diameter`, so the default 30 does nothing and 0 does nothing either. There is no automatic estimation.
- **An empty CP Model runs `cpsam`**, not cyto2, whatever the placeholder says.
- **`min_size=15` pixels is hardcoded** (`pipeline.py:143`, `:299`) and not exposed in the interface. On a 6.25 nm/px grid that removes anything below roughly 27 nm equivalent diameter, and on a 25 nm/px grid, roughly 110 nm. If your smallest pores are near that limit, they were removed before the metrics were computed.

See [Cellpose Segmentation](../guide/step3-segmentation.md) and [Parameters](../reference/parameters.md).

If Cellpose is producing nothing on an image that clearly contains fenestrations, also confirm the upsampled image is what you think it is. A run at the wrong pixel scale changes the apparent size of every pore. See [The scale-domain question](scale-domain.md).

## napari appears frozen during CLAHE

There is no error text. The window stops repainting after you click **Run Upsampling** with `CLAHE (CPU)` selected, and the operating system may mark it as not responding.

**Cause.** The CLAHE path runs on the Qt thread rather than in a worker (`_widget.py:312`), so nothing repaints until it finishes. The deep-learning and Cellpose steps do use workers and stay responsive.

**Fix.** Wait. It is not crashed. The window returns when the `Upsampled AFM` layer appears and the button reads `Run Upsampling` again. The slow step is the cubic zoom at `pipeline.py:52`, and its cost grows with the square of the factor, so try a small factor first to gauge how long your scan takes.

## Padding size should be less than the corresponding input dimension

```text
RuntimeError: Padding size should be less than the corresponding input dimension
```

Appears inside a `Container DL Inference failed:` dialog during a **SwinIR** run.

**Cause.** Each tile is padded to a multiple of 16 (`inference.py:166-167`) even though SwinIR's window is 8 (`inference.py:71`). With the fixed tile size of 256 and a 32-pixel overlap the stride is 224, so an image height of the form 224k + 8 leaves a final tile 8 rows tall that needs 8 rows of reflect padding, which torch rejects.

The first failing heights are **232, 456, 680 and 904**. Of the heights from 8 to 4096, 145 fail. HAT is not affected.

**Fix.** Switch Method to **HAT**, or crop the scan to a safe size. 256, 512, 1024 and 2048 are all safe.

## Batch finished but the workbook has fewer images than expected

```text
Successfully processed 24 images.
```

The dialog reports the number of files found, but `batch_results.xlsx` contains fewer distinct `Image_Name` values.

**Cause.** An image where Cellpose found nothing produces an empty table with zero rows, and `pd.concat` at `pipeline.py:437` contributes nothing for it. The image is absent from the workbook rather than present with a zero.

**Fix.** Use the per-image mask files as the complete record. One `<stem>_mask.tif` is written for every image (`pipeline.py:426`) before quantification:

```bash
ls "$OUTDIR"/*_mask.tif | wc -l
```

Compare that count with the distinct `Image_Name` values in the workbook. Every mask file without rows is a zero-detection image. Open it in napari to decide whether the scan was genuinely empty or the segmentation failed, and record it explicitly rather than letting it drop out of your *n*. Full detail in [Known issues](known-issues.md#1-batch-drops-images-where-cellpose-finds-nothing).
