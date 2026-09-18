# Troubleshooting

Error messages you can hit while installing or running FenestRA, with the cause and the fix for each. Sections are organized by the text you see, so you can search this page for the string in your dialog or terminal. The last four sections cover failures that happen before the plugin ever loads: Qt bindings, Windows application control, and building or opening the all-in-one image.

If the run completed but the numbers look wrong, this is the wrong page. Go to [Known issues](known-issues.md) or [The scale-domain question](scale-domain.md).

Line numbers on this page refer to FenestRA 0.3.0. Where 0.3.0 changed the behaviour, the older symptom is given as well, so an earlier install is still recognisable.

## The plugin does not appear under Plugins

There is no error text. The napari **Plugins** menu has no `FenestRA Pipeline` entry.

**Check the environment first.** napari has to be the one inside the environment where you installed the plugin:

```bash
conda activate fenestra-env
which napari
pip show napari-fenestra
python -c "import fenestra; print(fenestra.__file__, fenestra.__version__)"
```

If `pip show` reports nothing, the plugin is installed somewhere else. Install it into the active environment.

From 0.3.0 `fenestra.__version__` reads the installed distribution metadata, so it agrees with `pip show` and is worth quoting in a bug report. Before 0.3.0 it was a hardcoded string that said `0.0.1` in every release; `0.0.0+unknown` means the package is being imported from a source tree that was never installed.

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

**Cause.** AFMReader is not installed in the active environment. `pipeline.py:16-19` catches the `ImportError` at import time and prints a warning to the terminal, so the plugin still loads and the failure only surfaces when you open a file:

```text
Warning: AFMReader not found or could not be imported.
```

**Fix.** AFMReader is on PyPI and installs with plain `pip`:

```bash
pip install "pySPM<0.6.3" "AFMReader==0.0.7"
```

`pySPM` is held below 0.6.3 because 0.6.3 requires NumPy 2. Git is not involved, so the old
`ERROR: Cannot find command 'git'` on Windows cannot happen here any more.

As of 0.3.0 `setup.cfg` declares `AFMReader>=0.0.7` and `pySPM<0.6.3` in `install_requires`, so a
fresh `pip install napari-fenestra` pulls the reader in by itself. An environment built before
0.3.0, when the instructions used `pip install git+https://github.com/AFM-SPM/AFMReader.git`, can
still be missing it — run the command above into that environment.

See [Host environment](../install/host-environment.md).

**If AFMReader is installed** and the message after `Failed to load JPK file:` is something else, that text comes from AFMReader itself. The plugin always requests the `height_trace` channel (`pipeline.py:57-61`), so a file that does not carry that channel fails here.

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

`pipeline.py:221` and `:354` choose the device from `torch.cuda.is_available()`, so this puts Cellpose on the CPU.

!!! warning
    With the Singularity engine the container subprocess inherits the same environment, so the deep-learning step also falls back to CPU and becomes very slow. The Local (bundled) engine behaves the same way: it passes the whole environment through, minus `PYTHONPATH` and `PYTHONHOME` (`pipeline.py:134`). If you need both, run the upsampling step in a session with the GPU visible and the segmentation step in a session without it.

    The Docker engine is the exception: `_build_dl_cmd` builds `docker run` with no `-e` flags (`pipeline.py:158-161`), so no host variable is forwarded, and the image sets `ENV CUDA_VISIBLE_DEVICES=0` internally (`containers/Dockerfile:63`). Hiding the GPU from napari therefore does not hide it from a Docker-engine inference run.

## Container DL Inference failed

```text
Container DL Inference failed: <everything the container wrote to stderr>
```

Raised whenever the inference process exits with a non-zero status, at `pipeline.py:184`. Since 0.3.0 that is one line reached by both the interactive and the batch path, and by all three engines, so the message no longer depends on which one you used. The dialog contains the process's full stderr, so scroll to the **last few lines**, which carry the actual Python traceback.

**Check these before reading the stderr:**

| Setting | What to confirm |
|---|---|
| DL Model | The `.pth` file exists on the host. If not, the plugin stops earlier with `Model path is invalid.` (`_widget.py:384`). |
| Singularity (.sif) | The file exists. If not, the plugin stops earlier with `Singularity container path is invalid.` (`_widget.py:390`). |
| Docker Tag | Not validated by the plugin. A typo reaches docker and comes back as a docker error. |
| DL backend (Local) | Not taken from the box, which is disabled on this engine. The interpreter is `FENESTRA_DL_PYTHON`, or `/opt/venv-dl/bin/python` (`pipeline.py:98`, `:115`). |
| Engine | Matches what is installed on your machine. `Local (bundled)` exists only inside the all-in-one image. |
| Method | HAT sends `--arch hat`, SwinIR sends `--arch swinir` (`_widget.py:380`). The checkpoint has to match the architecture. |

**Then read the stderr:**

| Last lines of stderr | Cause | Fix |
|---|---|---|
| `can't open file '/opt/python'` | The Docker image predates the ENTRYPOINT fix | Rebuild it. See the next section. |
| `Error(s) in loading state_dict`, with size mismatches | The checkpoint is a different architecture than the Method dropdown selected, or it uses `embed_dim=96` and `depths=[6]*4`, which the plugin cannot build (`inference.py:47-70`, `strict=True` at `:87`) | Switch the Method to match the checkpoint, or use an `embed_dim=180` checkpoint. See [Known issues](known-issues.md#7-only-embed_dim-180-checkpoints-load). |
| `ImportError: HAT arch could not be imported` | The container does not have HAT at `/opt/HAT` | Rebuild the container from the recipe in `containers/`. See [Container backend](../install/container-backend.md). |
| `SwinIR arch could not be imported` | The container has neither basicsr's SwinIR nor `/opt/SwinIR` | Rebuild the container. |
| `Padding size should be less than the corresponding input dimension` | SwinIR tiling on an unlucky image height | See the SwinIR section below. |

**To rule out the command itself**, run the argv checks that ship with the repository, from a checkout and inside the activated environment. They build the command for each engine and assert its shape, without launching anything:

```bash
conda activate fenestra-env
python tests/test_dl_cmd.py
```

Six checks pass in a healthy checkout. A failure there means the command is wrong before it ever reaches your container runtime.

**A missing runtime looks different.** If the engine's own executable cannot be launched at all, the failure comes from `subprocess.run` before any inference starts:

```text
[Errno 2] No such file or directory: 'docker'
```

for example when Docker is selected on a machine that only has Apptainer. Both paths now report it the same way, under a dialog titled **DL Error** for an interactive run (`_widget.py:442`) and **Batch Error** for a batch run (`_widget.py:625`).

!!! note "If your message starts with `Background thread error:`"

    You are on an install older than 0.3.0. The interactive path used to wrap the whole subprocess
    block and re-raise everything with that prefix, so a plain non-zero container exit and a
    missing `docker` binary were indistinguishable at a glance, while the batch path reported a
    bare `FileNotFoundError` for the same problem. The wrapper is gone; the text that used to
    follow the prefix is now the whole message, and the table above still applies to it.

## Docker cannot open the file /opt/python

```text
python: can't open file '/opt/python': [Errno 2] No such file or directory
```

**Cause.** Your Docker image was built from a checkout where `containers/Dockerfile` still ended with `ENTRYPOINT ["python"]`. The plugin appends its own `python /opt/dl_project/scripts/inference.py ...` as the command (`pipeline.py:145-146`), and Docker concatenates ENTRYPOINT and CMD, so the container runs `python python /opt/dl_project/scripts/inference.py` and Python treats the literal word `python` as the script path.

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

This concerns the deep-learning backend image, `livrvub/dl-upsampling`. The all-in-one image runs
the deep-learning step in-process with the **Local (bundled)** engine, so it launches no second
container and cannot produce this error.

## The bundled deep-learning environment was not found

```text
The bundled deep-learning environment was not found at /opt/venv-dl/bin/python.
The Local engine exists only inside the FenestRA all-in-one container.
Set FENESTRA_DL_PYTHON, or choose the Singularity or Docker engine.
```

New in 0.3.0, raised before anything is launched (`pipeline.py:117-121`) when **Engine** is set to
`Local (bundled)` and the interpreter it names does not exist.

**Cause.** The Local engine runs `inference.py` as a plain subprocess in a second Python
environment on the same filesystem, rather than inside a container. That environment is built by
`containers/Dockerfile.allinone` and lives at `/opt/venv-dl/bin/python`, which is what the image
sets `FENESTRA_DL_PYTHON` to. On a native install nothing is there.

**Fix.** Choose **Singularity** or **Docker**, which is the right answer on a native install. If
you did build an equivalent environment yourself, point `FENESTRA_DL_PYTHON` at its interpreter
before launching napari. See [The all-in-one container](../install/all-in-one.md).

!!! warning "The bundled environment is not the reference stack"

    `/opt/venv-dl` runs torch 2.1.2 with torchvision 0.16.2, because the reference stack's torch
    1.14 wheels carry no PTX and will not start on a GPU newer than sm_86. The validated backend
    is still `containers/dl_upsampling.def`, torch 1.14 on `nvcr.io/nvidia/pytorch:23.01-py3`, and
    numbers intended for publication should come from that container rather than from the
    all-in-one image.

## No output generated from DL Upsampling

```text
No output generated from DL Upsampling.
```

Raised at `pipeline.py:188` when the inference process exited successfully but no `*.tif*` file appeared in the output directory.

**Cause 1: the input is a uniform image.** `inference.py:101-103` prints a warning and returns without writing when the scan has no height variation at all, and the script still exits 0:

```text
Warning: temp_in.tif is a uniform image. Skipping.
```

Check the raw layer in napari. A flat or failed scan produces this.

**Cause 2: the output bind mount did not resolve.** With the Singularity and Docker engines the plugin mounts the host temp directory into the container as `/tmp_out` (`pipeline.py:140`, and the `-v` flags at `:161`). On Windows and macOS, Docker Desktop only shares directories you have allowed in its file-sharing settings, and the plugin uses the system temporary directory. Add that location to the shared paths. The Local (bundled) engine passes real paths and mounts nothing (`pipeline.py:123-130`), so this cause does not apply to it.

**Not a cause since 0.3.0: stale output.** The interactive path reuses one temporary directory for the whole napari session (`_widget.py:49`, `:419`) and loads the first `*.tif*` it finds, but `_run_dl_inference` now deletes any `*.tif*` still sitting there before it launches (`pipeline.py:179-180`), so a run that writes nothing raises this error instead of quietly loading an earlier result. On 0.2.11 and earlier it did load the earlier result; if you are on an older install and are not certain which result you are looking at, restart napari and run once. The batch path was never affected, because it creates a fresh temporary directory per image (`pipeline.py:468`).

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

- **Diameter is a rescale factor, not a size.** Cellpose 4 computes `image_scaling = 30 / diameter`, so the default 30 does nothing and 0 does nothing either. There is no automatic estimation. This is unchanged in 0.3.0; only the label was corrected, from `Diameter (0=auto)` to `Diameter (30 = no rescale)` (`_widget.py:208`). If your interface still offers an "auto" diameter, it is describing behaviour Cellpose 4 does not have.
- **An empty CP Model runs `cpsam`**, not cyto2. Also unchanged: the plugin passes no `model_type` at all now (`pipeline.py:244`, `:369`), where it used to pass `model_type="cyto2"` and Cellpose 4 accepted and ignored it. The model that ran was always `cpsam`. Before 0.3.0 the placeholder read *"Leave empty for cyto2"* and was wrong; it now reads *"Leave empty for the Cellpose 4 default (cpsam)"* (`_widget.py:193`). If you have older results and are not sure which model produced them, an empty box means `cpsam` in every version.
- **`min_size=15` pixels is hardcoded** (`pipeline.py:253`, `:378`) and not exposed in the interface. On a 6.25 nm/px grid that removes anything below roughly 27 nm equivalent diameter, and on a 25 nm/px grid, roughly 110 nm. If your smallest pores are near that limit, they were removed before the metrics were computed.

See [Cellpose Segmentation](../guide/step3-segmentation.md) and [Parameters](../reference/parameters.md).

If Cellpose is producing nothing on an image that clearly contains fenestrations, also confirm the upsampled image is what you think it is. A run at the wrong pixel scale changes the apparent size of every pore. See [The scale-domain question](scale-domain.md).

## napari appears frozen during CLAHE

There is no error text. The window stops repainting after you click **Run Upsampling** with `CLAHE (CPU)` selected, and the operating system may mark it as not responding.

**Cause.** The CLAHE path runs on the Qt thread rather than in a worker (`_widget.py:370`), so nothing repaints until it finishes. The deep-learning and Cellpose steps do use workers and stay responsive.

**Fix.** Wait. It is not crashed. The window returns when the `Upsampled AFM` layer appears and the button reads `Run Upsampling` again. The slow step is the cubic zoom at `pipeline.py:85`, and its cost grows with the square of the factor, so try a small factor first to gauge how long your scan takes.

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

**Cause.** An image where Cellpose found nothing produces an empty table with zero rows, and `pd.concat` at `pipeline.py:517` contributes nothing for it. The image is absent from the workbook rather than present with a zero.

**Fix.** Use the per-image mask files as the complete record. One `<stem>_mask.tif` is written for every image (`pipeline.py:506`) before quantification:

```bash
ls "$OUTDIR"/*_mask.tif | wc -l
```

Compare that count with the distinct `Image_Name` values in the workbook. Every mask file without rows is a zero-detection image. Open it in napari to decide whether the scan was genuinely empty or the segmentation failed, and record it explicitly rather than letting it drop out of your *n*. Full detail in [Known issues](known-issues.md#1-batch-drops-images-where-cellpose-finds-nothing).

## No Qt bindings could be found

```text
qtpy.QtBindingsNotFoundError: No Qt bindings could be found
ImportError: Failed to import Qt bindings. We found following Qt bindings installed: pyqt6=6.11.0.
  pyqt6: ImportError: DLL load failed while importing QtWidgets:
         The specified procedure could not be found.
```

Seen on Windows, when launching `napari` in an environment where PyQt6 is clearly installed.

**Cause.** Read the last line carefully: the specified **procedure** could not be found, not the
specified *module*. That distinction is the whole diagnosis. A missing Visual C++ runtime produces
"module". "Procedure" means a `Qt6*.dll` *was* loaded but does not export a symbol PyQt6 expects —
i.e. `PyQt6`, `PyQt6-Qt6` and `PyQt6_sip` are at mismatched versions, or an older `Qt6Core.dll`
somewhere on `PATH` is shadowing the one PyQt6 ships. Anaconda puts its own `Library\bin` on
`PATH`, which is the usual source.

`napari[all]` resolves to `PyQt6>6.5` with **no upper bound**, so two people installing on
different days get different Qt versions. Since 0.3.0 the install instructions pin
`PyQt6==6.11.0` for exactly this reason ([Host environment](../install/host-environment.md)); an
environment created from the earlier, unpinned instructions is the usual place this appears.

**Diagnose.** In the activated environment:

```bat
pip list | findstr /I "PyQt6 PySide qtpy napari"
where Qt6Core.dll
```

`PyQt6` and `PyQt6-Qt6` must be the *same* version, and `where Qt6Core.dll` should find nothing
outside `...\site-packages\PyQt6\Qt6\bin\`.

**Fix.** Reinstall a matched set:

```bat
pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
pip cache purge
pip install "PyQt6==6.11.0" "PyQt6-Qt6==6.11.0"
```

If it still fails, switch the binding entirely — napari supports PySide6 equally well, and it
sidesteps both the sip mismatch and the DLL shadowing:

```bat
pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
pip install "napari[pyside6]"
```

FenestRA itself needs no change either way: it goes through `qtpy`.

**Or stop fighting it.** The [all-in-one container](../install/all-in-one.md) runs Qt on Linux
inside the image, against a pinned PyQt6 6.11.0, so this class of failure cannot occur.

## Application control policies have blocked this file

```text
ImportError: DLL load failed while importing _solve_toeplitz:
    This file is blocked by application control policies.
```

The name of the blocked file varies — `_solve_toeplitz` (SciPy), `_ufuncs`, a NumPy or PyTorch
`.pyd`. The message is the constant part. On a non-English Windows it is localised, e.g. Polish
*"Zasady kontroli aplikacji zablokowały ten plik."*

**Cause.** Not a Python problem at all. Windows refused to load the file. A code-integrity policy —
**Smart App Control**, **WDAC**, or **AppLocker** DLL rules — is blocking unsigned DLLs under
`C:\Users\`, which is exactly where conda and pip put every compiled scientific package. SciPy is
merely the first one that happened to be imported; NumPy, PyTorch and PyQt6 are all equally
blocked.

**Diagnose.** In PowerShell:

```powershell
# Smart App Control: 0 = off, 1 = enforced, 2 = evaluation
Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" -Name VerifiedAndReputablePolicyState -ErrorAction SilentlyContinue

# WDAC / Device Guard
Get-CimInstance -Namespace root\Microsoft\Windows\DeviceGuard -ClassName Win32_DeviceGuard |
  Select-Object CodeIntegrityPolicyEnforcementStatus, UsermodeCodeIntegrityPolicyEnforcementStatus

# AppLocker — names the exact blocked file
Get-WinEvent -LogName "Microsoft-Windows-AppLocker/EXE and DLL" -MaxEvents 20 |
  Select-Object TimeCreated, Id, Message | Format-List
```

**Fix, if it is Smart App Control.** Windows Security → App & browser control → Smart App Control →
Off.

!!! danger "Turning Smart App Control off is one-way"

    Once disabled, Windows will not let you re-enable it without reinstalling Windows. That is
    Microsoft's design. Decide before you click.

**Fix, if it is WDAC or AppLocker.** On a managed institutional laptop you cannot and should not
override this yourself. Your IT department must allow-list the environment directory for DLL
loading; send them the AppLocker event from above.

**If IT will not allow-list it,** use the [all-in-one container](../install/all-in-one.md). Code
integrity policies apply to the Windows host, not to the Linux filesystem inside the container, so
the problem disappears — and the container backend needs WSL 2 on Windows anyway.

## The browser shows a directory listing instead of napari

You opened `http://localhost:6080` and got a file index containing `vnc.html`, `vnc_lite.html` and
friends.

**Cause.** noVNC ships no `index.html`. The image writes one during the build; an image built
before that line existed does not have it.

**Fix.** Rebuild the image, or navigate to `http://localhost:6080/vnc.html?autoconnect=1` directly.

## failed to read dockerfile, transferring dockerfile: 2B

```text
#1 transferring dockerfile: 2B done
ERROR: failed to solve: failed to read dockerfile:
       open Dockerfile.allinone: no such file or directory
```

The file is plainly there, and `cat` prints it.

**Cause.** Linux, with Docker installed from the Canonical **snap**. The snap is confined and
cannot read `/media` or `/mnt` at all, so a repository on an external or secondary drive presents
an empty build context. The `2B` is the tell: that is an empty transfer, not a small file.

**Fix.** Grant the interface:

```bash
sudo snap connect docker:removable-media
```

or clone the repository somewhere under your home directory and build from there. Confirm which
Docker you have with `readlink -f "$(command -v docker)"`; if the answer is `/usr/bin/snap`, this
is your problem.
