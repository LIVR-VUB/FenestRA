# 5 - Verify the install

Work down this checklist once, on any scan, to confirm that a `.jpk-qi-image` goes in and a CSV
comes out. It uses the **CLAHE (CPU)** method throughout, so it needs no model weights and no
container. Each item says what you should see and what it means if you do not.

## The checklist

- [ ] **napari starts.**

    ```bash
    conda activate fenestra-env
    napari
    ```

    An empty viewer window opens. If the shell reports that `napari` is not a command, the
    environment is not active or `napari[all]` was never installed. If the process exits with a Qt
    error, you have `napari` without its Qt backend: reinstall with the `[all]` extra.

- [ ] **The plugin is listed.** Open the **Plugins** menu and choose **FenestRA Pipeline**. A dock
  appears on the right with five numbered panels, `1. Input Data` down to `5. Batch Analysis`.

    If the entry is missing, the plugin is installed into a different environment from the one
    napari is running in. Check with `pip show napari-fenestra` in the same active environment,
    then restart napari.

- [ ] **A scan loads and reports its scale.** In panel 1, click **Load JPK.qi-image** and pick a
  file. The dialog filters to `*.jpk` and `*.jpk-qi-image`.

    You should see a `Raw AFM` layer in the layer list, drawn with the magma colormap, and the
    label under the button should read two lines:

    ```text
    Size: (512, 512)
    Scale: 100.00 nm/px
    ```

    Your numbers will differ. The **Scale** value is the one that matters: it comes from the file
    and it is what converts every later measurement into nanometers. Note it down.

    A dialog reading `Could not load JPK:` means AFMReader is missing, or the file has no
    `height_trace` channel. That channel is the one the loader requests.

- [ ] **CLAHE upsampling produces a layer.** In panel 2, set **Method** to `CLAHE (CPU)` and leave
  **Factor** at `4`. Click **Run Upsampling**.

    The button changes to `Upsampling in progress...`, then a second image layer named
    `Upsampled AFM` appears, four times larger in pixels and aligned on top of `Raw AFM`.

    If napari stops responding while it works, it has not crashed. CLAHE runs on the interface
    thread, so the window is frozen until the computation finishes. On a large scan that can take
    a while.

    Keep **Factor** at 4 for this check. The viewer draws the upsampled layer at a fixed quarter
    scale, so any other factor makes the two layers look aligned when they are not.

- [ ] **Cellpose produces masks.** In panel 3, leave **CP Model** empty and the three numbers at
  their defaults (`30.00`, `0.00`, `0.40`). Click **Run Cellpose**.

    The button changes to `Segmenting...`, then a labels layer named `Cellpose Masks` appears with
    each detected pore in its own color. Toggle `Upsampled AFM` on and off to see whether the
    masks land on the pores.

    Timing varies a lot here, because segmentation uses the GPU when PyTorch can see one and the
    CPU otherwise. A `Cellpose Error` dialog is a real failure; see
    [Troubleshooting](../caveats/troubleshooting.md).

    With **CP Model** empty you get Cellpose 4's default model, `cpsam`, not the `cyto2` the
    placeholder text names. This is expected, and explained in
    [Segmentation](../guide/step3-segmentation.md).

- [ ] **Quantify writes a CSV.** In panel 4, click **Quantify Fenestrations**. A save dialog opens
  with the filename `fenestration_metrics.csv`. Choose a location and save.

    A message box then reports the count and the porosity:

    ```text
    Saved 214 fenestrations.
    Overall Porosity: 4.31%
    ```

    Open the CSV. It should have these columns, in this order:

    ```text
    Label, Area_nm2, Perimeter_nm, Equivalent_Diameter_nm,
    Equivalent_Diameter_Upsampled_Pixels, Equivalent_Diameter_Raw_Pixels, Eccentricity
    ```

    There is no porosity column in the single-image CSV. Porosity appears only in that message
    box. Batch runs write it into the spreadsheet instead.

    To confirm the units are right, take any row and check that `Equivalent_Diameter_nm` equals
    `Equivalent_Diameter_Upsampled_Pixels` multiplied by your **Scale** value divided by the
    factor you upsampled with.

!!! warning "Do not touch the Method dropdown between the last two steps"

    **Quantify Fenestrations** reads the current value of **Method** and **Factor**, not the ones
    that were used to produce the image on screen. If you run CLAHE at factor 2 and then switch
    the dropdown to `HAT` before clicking Quantify, every diameter is reported half its true size
    with no warning. Run, then quantify, then change settings.

## Verifying the container backend

Do this once you hold model weights. Test the container on its own first, so that a GPU
passthrough problem does not look like a model problem.

=== "Linux (Apptainer)"

    ```bash
    singularity exec --nv dl_upsampling.sif python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    ```

    This prints the container's own PyTorch version, which is older than the one on your host, and
    `True` if the GPU is visible inside the container. `False` means `--nv` found no driver to
    bind.

=== "Windows / macOS (Docker)"

    ```bash
    docker run --rm --gpus all livrvub/dl-upsampling:latest -c "import torch; print(torch.cuda.is_available())"
    ```

    Note the absent `python`: the image already has `python` as its entrypoint, so the arguments
    you pass are appended to it. That is also the reason the plugin's own Docker command currently
    fails, since it supplies a second `python`. See
    [Container backend](container-backend.md).

    This form assumes the shipped `ENTRYPOINT ["python"]` is unchanged. If you fixed the Docker
    path by editing the Dockerfile rather than the plugin, put `python` back into the command
    above or the container has nothing to execute.

Then run it through the plugin: set **Method** to `HAT` or `SwinIR`, fill in **DL Model** with your
`.pth`, set **Engine** to match the image you built, and click **Run Upsampling**. Success looks
identical to the CLAHE step, an `Upsampled AFM` layer at four times the pixel size.

A dialog containing `Container DL Inference failed:` carries the container's own error output after
the colon. Read that text: it distinguishes a wrong-shape checkpoint from a missing GPU from a
mount that the engine refused.

## When something fails

[Troubleshooting](../caveats/troubleshooting.md) lists the failures by symptom.
[Known issues](../caveats/known-issues.md) covers the ones that are behaving as the current code
was written, rather than as you would expect.
