#!/usr/bin/env python3
"""
AFM Super-Resolution Inference Script

Upsamples raw AFM images (.tif) using pre-trained HAT or SwinIR models.
Automatically handles arbitrary image sizes by dynamically padding to the 
model's required window size, then cropping the output.

Usage:
  python inference.py --input /path/to/raw --output /path/to/save --model_path /path/to/best_model.pth --arch hat
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

# Modify this if your model arch files move
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def build_model(arch: str, model_path: str, device: str) -> torch.nn.Module:
    """Build and load the model weights."""
    arch = arch.lower()
    
    if arch == 'hat':
        try:
            # Try specific path directly bypassing package __init__ (avoids basicsr rgb2ycbcr error)
            import importlib.util
            spec = importlib.util.spec_from_file_location('hat_arch', '/opt/HAT/hat/archs/hat_arch.py')
            hat_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hat_module)
            HAT = hat_module.HAT
        except Exception as e:
            try:
                # Fallback to pip-installed version if local
                from hat.archs.hat_arch import HAT
            except ImportError:
                raise ImportError(f"HAT arch could not be imported. Original error: {e}")
                
        model = HAT(
            upscale=4, in_chans=1, img_size=32, window_size=16,
            compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
            overlap_ratio=0.5, img_range=1., depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
            upsampler='pixelshuffle', resi_connection='1conv'
        )
        window_size = 16
    elif arch == 'swinir':
        try:
            from basicsr.archs.swinir_arch import SwinIR
        except ImportError:
            try:
                sys.path.insert(0, '/opt/SwinIR')
                from models.network_swinir import SwinIR
            except ImportError:
                raise ImportError("SwinIR arch could not be imported.")
                
        model = SwinIR(
            upscale=4, in_chans=1, img_size=32, window_size=8,
            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle',
            resi_connection='1conv'
        )
        window_size = 8
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Load weights
    print(f"Loading {arch.upper()} weights from: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Handle different save formats
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
    elif 'ema' in state_dict:
        state_dict = state_dict['ema']
        
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device), window_size


def process_image(img_path: Path, out_path: Path, model: torch.nn.Module, 
                 window_size: int, device: str, tile_size: int = 0):
    """Upsample a single image and save the result."""
    
    # 1. Load image and determine physical data range
    img = tifffile.imread(str(img_path)).astype(np.float32)
    vmin, vmax = img.min(), img.max()
    
    # Skip extreme case of uniform image
    if vmax - vmin < 1e-8:
        print(f"Warning: {img_path.name} is a uniform image. Skipping.")
        return
        
    # Scale to [0, 1] — model expects this
    img_norm = (img - vmin) / (vmax - vmin)
    
    # Convert to tensor: [1, 1, H, W]
    tensor_img = torch.from_numpy(img_norm[None, None, ...]).to(device)
    _, _, h, w = tensor_img.shape
    
    # 2. Dynamic Padding for arbitrary sizes
    # Height and Width must be perfectly divisible by the model's window_size
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        tensor_img = F.pad(tensor_img, (0, pad_w, 0, pad_h), mode='reflect')
        
    # 3. Model Inference
    with torch.no_grad():
        if tile_size > 0:
            # Memory-safe tiled inference for massive images
            tensor_out = tile_inference(model, tensor_img, tile_size, scale=4)
        else:
            # Full image push (requires large GPU)
            tensor_out = model(tensor_img)
            
    # 4. Remove padding from the upscaled output
    # Since we upscaled by 4x, the padded pixels on the output are 4x larger
    if pad_h > 0:
        tensor_out = tensor_out[:, :, : -(pad_h * 4), :]
    if pad_w > 0:
        tensor_out = tensor_out[:, :, :, : -(pad_w * 4)]
        
    # 5. Reverse normalization and save
    out_npy = tensor_out.squeeze().cpu().numpy()
    out_npy = np.clip(out_npy, 0, 1)
    
    # Restore physical nanometer/micrometer scale
    out_restored = out_npy * (vmax - vmin) + vmin
    
    tifffile.imwrite(str(out_path), out_restored.astype(np.float32))


def tile_inference(model: torch.nn.Module, img: torch.Tensor, tile_size: int, scale: int = 4) -> torch.Tensor:
    """Perform overlapping tile inference if the GPU runs out of memory."""
    b, c, h, w = img.shape
    overlap = 32
    stride = tile_size - overlap
    
    out_h, out_w = h * scale, w * scale
    output = torch.zeros(b, c, out_h, out_w, device=img.device)
    weights = torch.zeros(b, c, out_h, out_w, device=img.device)
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # The tile itself might need padding if it gets cut off at the edge
            tile = img[:, :, y:y_end, x:x_end]
            
            curr_h, curr_w = tile.shape[2], tile.shape[3]
            # Assumes window_size 16 is safe for both HAT and SwinIR (since 16 is multiple of 8)
            pad_h = (16 - curr_h % 16) % 16
            pad_w = (16 - curr_w % 16) % 16
            
            if pad_h > 0 or pad_w > 0:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                
            with torch.no_grad():
                tile_out = model(tile)
                
            if pad_h > 0: tile_out = tile_out[:, :, :-(pad_h * scale), :]
            if pad_w > 0: tile_out = tile_out[:, :, :, :-(pad_w * scale)]
            
            out_y, out_x = y * scale, x * scale
            out_y_end, out_x_end = out_y + tile_out.shape[2], out_x + tile_out.shape[3]
            
            output[:, :, out_y:out_y_end, out_x:out_x_end] += tile_out
            weights[:, :, out_y:out_y_end, out_x:out_x_end] += 1
            
    return output / weights.clamp(min=1)


def main():
    parser = argparse.ArgumentParser(description="AFM Arbitrary Size Inference")
    parser.add_argument('--input', type=str, required=True, help="Input file or folder")
    parser.add_argument('--output', type=str, required=True, help="Output folder")
    parser.add_argument('--model_path', type=str, required=True, help="Path to best_model_ema.pth weights")
    parser.add_argument('--arch', type=str, default='hat', choices=['hat', 'swinir'])
    parser.add_argument('--tile_size', type=int, default=0, help="Set to 256 if you get OOM errors")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Executing inference on: {device.upper()}")

    # Output dir setup
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Input files
    in_path = Path(args.input)
    if in_path.is_file():
        files = [in_path]
    else:
        files = list(in_path.glob("*.tif")) + list(in_path.glob("*.tiff"))
        
    if not files:
        print(f"No tiff files found in {in_path}")
        return

    # Build model (loads weights to GPU)
    model, window_size = build_model(args.arch, args.model_path, device)
    
    print(f"Found {len(files)} images to upsample.")
    print(f"Architecture: {args.arch.upper()} | Dynamic Padding Window: {window_size}")
    
    for f in tqdm(files, desc="Upsampling"):
        out_f = out_dir / f"{f.stem}_SR4x.tif"
        process_image(f, out_f, model, window_size, device, args.tile_size)
        
    print("\nInference Complete!")


if __name__ == "__main__":
    main()
