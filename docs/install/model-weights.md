# 4 - Model weights

This page covers the trained checkpoints the **HAT** and **SwinIR** methods need, where the plugin
expects to find them, and which checkpoints it can actually load.

!!! info "Pre-publication notice"

    The fine-tuned HAT, SwinIR, and custom Cellpose LSEC weights are **not public**. They are held
    back until the peer-reviewed manuscript is published, and will be released with it. This
    repository is the code and the container recipes.

    There is nothing to download. If you already hold the checkpoints, the rest of this page tells
    you how to point the plugin at them.

## What works without weights

The **CLAHE (CPU)** method needs no checkpoint, no container, and no GPU. It upsamples by cubic
interpolation and applies contrast equalization and unsharp masking. Everything downstream of it
works normally: Cellpose segmentation, the 4-pane grid, the CSV, and batch runs. Use it to learn
the interface and to verify the install.

CLAHE is an interpolation, not a super-resolution model. It recovers no detail that was not in the
scan. Treat it as a working path, not as a substitute for the trained models.

## Where the plugin looks for the file

In panel 2, with **Method** set to `HAT` or `SwinIR`, the **DL Model:** field takes the full path
to a single `.pth` file. The `...` button opens a file dialog filtered to `*.pth`.

The plugin mounts the *parent directory* of that file into the container, so the checkpoint must
sit somewhere your container engine is allowed to read. A path under your home directory is fine
for both engines on Linux.

The field arrives pre-filled with a developer's path from the machine the plugin was built on. It
is not a real location on your system. Replace it.

## Which checkpoints load

The architecture is fixed in code rather than read from the checkpoint, so only checkpoints whose
shapes match that fixed definition will load:

| | HAT | SwinIR |
|---|---|---|
| `upscale` | 4 | 4 |
| `in_chans` | 1 | 1 |
| `embed_dim` | 180 | 180 |
| `depths` | `[6, 6, 6, 6, 6, 6]` | `[6, 6, 6, 6, 6, 6]` |
| `num_heads` | `[6, 6, 6, 6, 6, 6]` | `[6, 6, 6, 6, 6, 6]` |
| `window_size` | 16 | 8 |
| `upsampler` | `pixelshuffle` | `pixelshuffle` |

A checkpoint saved as a plain state dict works, and so does one wrapped under a `model`, `params`,
or `ema` key. Those three are unwrapped automatically.

**The "small" model variants do not load.** A checkpoint trained with `embed_dim=96` and
`depths=[6, 6, 6, 6]` has different tensor shapes, so loading raises a `RuntimeError` inside the
container. The plugin surfaces the container's stderr in a dialog whose text contains:

```text
Container DL Inference failed: <stderr>
```

That is a real failure, not a warning. No layer is added and nothing is written.

!!! note "Strict loading is deliberate"

    Weights are loaded with `strict=True`, which refuses any checkpoint whose parameter names or
    shapes do not match the model being built. The alternative, loading non-strictly, would accept
    a mismatched checkpoint by leaving some layers at their random initialization. That produces
    no error and an image that still looks like an AFM scan, which you would then segment and
    measure.

    A loud `RuntimeError` at load time is the better outcome. Do not relax it.

## The Cellpose model is separate

Panel 3 has its own **CP Model:** field for a custom Cellpose checkpoint, and it is independent of
the `.pth` used here. Leaving it empty does not give you the model its placeholder text names. See
[Segmentation](../guide/step3-segmentation.md).

## One thing to settle before you run

The deep learning path is always x4, and the models were trained to invert a synthetic x4
degradation from a specific acquisition scale. The scale you acquired at therefore determines
whether the output is in the domain the network was trained on, and nothing in the plugin checks
it. Read [The scale-domain question](../caveats/scale-domain.md) before measuring anything you
intend to publish.

Next: [5 - Verify the install](verify.md).
