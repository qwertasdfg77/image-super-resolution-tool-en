#!/usr/bin/env python
"""
DLSS-like single-image super-resolution for NVIDIA CUDA GPUs.

This is not real DLSS: DLSS needs game-engine data such as motion vectors,
depth, and multiple frames. This script uses Real-ESRGAN, HAT, and ATD
super-resolution models plus automatic denoise/sharpen post-processing to get a similar
"AI upscaled and sharpened" look for still images.
"""

from __future__ import annotations

import argparse
import html
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

warnings.filterwarnings("ignore", message="torch.meshgrid:.*indexing", category=UserWarning)


MODEL_SPECS = {
    "photo": {
        "name": "RealESRGAN_x4plus",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "family": "gan",
        "display": "Real-ESRGAN x4 General Photo Model",
        "num_block": 23,
    },
    "anime": {
        "name": "RealESRGAN_x4plus_anime_6B",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "family": "gan",
        "display": "Real-ESRGAN x4 Anime and Illustration Model",
        "num_block": 6,
    },
    "hat": {
        "name": "Real_HAT_GAN_sharper",
        "url": "https://huggingface.co/Acly/hat/resolve/main/Real_HAT_GAN_sharper.pth",
        "fallback_urls": [
            "https://hf-mirror.com/Acly/hat/resolve/main/Real_HAT_GAN_sharper.pth",
        ],
        "scale": 4,
        "family": "transformer",
        "display": "HAT x4 High-Quality Super-Resolution Model",
        "loader": "spandrel",
    },
    "atd": {
        "name": "003_ATD_SRx4_finetune",
        "url": "https://drive.google.com/uc?export=download&confirm=1&id=1J9kR9OyrOxtJ5Ygbr_W116BLBwnD4VNL",
        "scale": 4,
        "family": "transformer",
        "display": "ATD x4 Official Super-Resolution Model",
        "loader": "spandrel",
        "preferred_precision": "fp32",
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
OUTPUT_FORMAT_EXTENSIONS = {
    "auto": None,
    "png": ".png",
    "jpeg": ".jpg",
    "webp": ".webp",
}

TORCH = None
NN = None
F = None
SPANDREL_MODEL_LOADER = None
SPANDREL_IMAGE_MODEL_DESCRIPTOR = None


@dataclass
class GpuProfile:
    name: str = "CPU"
    capability: str = "n/a"
    vram_gb: float = 0.0
    free_gb: float = 0.0
    multiprocessors: int = 0


@dataclass
class RuntimeTuning:
    tile: int
    tile_pad: int
    precision: str
    channels_last: bool
    memory_fraction: float | None


def import_torch():
    """Import torch lazily so --help still works before torch is installed."""
    global TORCH, NN, F
    if TORCH is not None:
        return TORCH

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install the CUDA build first, for example:\n"
            "  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        ) from exc

    TORCH = torch
    NN = nn
    F = functional
    return torch


def import_spandrel():
    """Import Spandrel lazily so Real-ESRGAN can still run without it."""
    global SPANDREL_MODEL_LOADER, SPANDREL_IMAGE_MODEL_DESCRIPTOR
    if SPANDREL_MODEL_LOADER is not None:
        return SPANDREL_MODEL_LOADER, SPANDREL_IMAGE_MODEL_DESCRIPTOR

    try:
        from spandrel import ImageModelDescriptor, ModelLoader
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Spandrel is not installed. Click the app's install/check environment button first, or run:\n"
            "  python -m pip install spandrel>=0.4.2"
        ) from exc

    SPANDREL_MODEL_LOADER = ModelLoader
    SPANDREL_IMAGE_MODEL_DESCRIPTOR = ImageModelDescriptor
    return ModelLoader, ImageModelDescriptor


def cuda_profile(device: str) -> GpuProfile:
    torch = import_torch()
    if device != "cuda":
        return GpuProfile()
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU was required, but PyTorch cannot see CUDA.\n"
            "Check the NVIDIA driver and install the CUDA PyTorch build."
        )

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return GpuProfile(
        name=torch.cuda.get_device_name(0),
        capability=f"{major}.{minor}",
        vram_gb=total_bytes / (1024**3),
        free_gb=free_bytes / (1024**3),
        multiprocessors=getattr(props, "multi_processor_count", 0),
    )


def choose_runtime_tuning(profile: GpuProfile, spec: dict, usage: str) -> RuntimeTuning:
    vram = profile.vram_gb
    usage = "balanced" if usage == "auto" else usage

    if spec.get("family") == "transformer":
        tiers = [
            (4.5, 48),
            (6.5, 56),
            (8.5, 64),
            (12.5, 96),
            (16.5, 128),
            (24.5, 160),
            (999.0, 192),
        ]
        floor_tile = 48
    else:
        tiers = [
            (4.5, 192),
            (6.5, 256),
            (8.5, 384),
            (12.5, 512),
            (16.5, 640),
            (24.5, 768),
            (999.0, 896),
        ]
        floor_tile = 128

    tile = tiers[-1][1]
    tier_index = len(tiers) - 1
    for index, (limit, candidate) in enumerate(tiers):
        if vram <= limit:
            tile = candidate
            tier_index = index
            break

    if usage == "conservative" and tier_index > 0:
        tile = tiers[tier_index - 1][1]
    elif usage == "performance" and tier_index < len(tiers) - 1:
        tile = tiers[tier_index + 1][1]

    tile = max(floor_tile, tile)
    if spec.get("family") == "transformer":
        tile_pad = max(12, min(32, tile // 4))
    else:
        tile_pad = max(16, min(48, tile // 6))

    if usage == "conservative":
        memory_fraction = 0.72
    elif usage == "performance":
        memory_fraction = 0.94
    elif vram <= 8.5:
        memory_fraction = 0.84
    elif vram <= 12.5:
        memory_fraction = 0.88
    else:
        memory_fraction = 0.91

    capability = 0.0 if profile.capability == "n/a" else float(profile.capability)
    if spec.get("preferred_precision"):
        precision = spec["preferred_precision"]
        channels_last = False
    elif spec.get("family") == "transformer":
        precision = "bf16" if capability >= 8.0 else "fp32"
        channels_last = False
    else:
        precision = "fp16" if capability >= 7.0 else "fp32"
        channels_last = True
    return RuntimeTuning(
        tile=tile,
        tile_pad=tile_pad,
        precision=precision,
        channels_last=channels_last,
        memory_fraction=memory_fraction,
    )


def apply_runtime_tuning(args, spec: dict, profile: GpuProfile) -> None:
    tuning = choose_runtime_tuning(profile, spec, args.gpu_usage)

    if args.tile == "auto":
        args.tile = tuning.tile
    else:
        args.tile = int(args.tile)

    if args.tile_pad == "auto":
        args.tile_pad = tuning.tile_pad
    else:
        args.tile_pad = int(args.tile_pad)

    if args.precision == "auto":
        args.precision = tuning.precision

    if args.channels_last == "auto":
        args.channels_last = tuning.channels_last
    else:
        args.channels_last = args.channels_last == "on"

    if args.gpu_memory_fraction == "auto":
        args.gpu_memory_fraction = tuning.memory_fraction
    else:
        args.gpu_memory_fraction = float(args.gpu_memory_fraction)

    if args.device == "cuda" and args.gpu_memory_fraction:
        torch = import_torch()
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)

    print(
        "Auto tuning: "
        f"model={spec.get('display', spec['name'])} | "
        f"tile={args.tile} | tile_pad={args.tile_pad} | "
        f"precision={args.precision} | "
        f"gpu_usage={args.gpu_usage} | "
        f"memory_limit={args.gpu_memory_fraction:.0%}"
    )


def make_layer(block: Callable, num_blocks: int, **kwargs):
    return NN.Sequential(*[block(**kwargs) for _ in range(num_blocks)])


def pixel_unshuffle(x, scale: int):
    b, c, h, w = x.size()
    if h % scale != 0 or w % scale != 0:
        raise ValueError(f"Input size {w}x{h} is not divisible by pixel-unshuffle scale {scale}.")
    x = x.view(b, c, h // scale, scale, w // scale, scale)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * scale * scale, h // scale, w // scale)


def build_rrdbnet(num_block: int, scale: int = 4):
    import_torch()

    class ResidualDenseBlock(NN.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
            super().__init__()
            self.conv1 = NN.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = NN.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = NN.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = NN.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = NN.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = NN.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(TORCH.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(TORCH.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(TORCH.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(TORCH.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(NN.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(NN.Module):
        def __init__(
            self,
            num_in_ch: int = 3,
            num_out_ch: int = 3,
            num_feat: int = 64,
            num_blocks: int = 23,
            num_grow_ch: int = 32,
            model_scale: int = 4,
        ):
            super().__init__()
            self.scale = model_scale
            first_in_ch = num_in_ch
            if model_scale == 2:
                first_in_ch *= 4
            elif model_scale == 1:
                first_in_ch *= 16

            self.conv_first = NN.Conv2d(first_in_ch, num_feat, 3, 1, 1)
            self.body = make_layer(RRDB, num_blocks, num_feat=num_feat, num_grow_ch=num_grow_ch)
            self.conv_body = NN.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = NN.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = NN.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = NN.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = NN.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = NN.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            if self.scale == 2:
                feat = pixel_unshuffle(x, scale=2)
            elif self.scale == 1:
                feat = pixel_unshuffle(x, scale=4)
            else:
                feat = x

            feat = self.conv_first(feat)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet(num_blocks=num_block, model_scale=scale)


def request_url(opener, url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "ImageSuperResolutionTool/1.0"})
    return opener.open(request, timeout=60)


def resolve_google_drive_warning(url: str, opener) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.lower() != "drive.google.com":
        return url

    with request_url(opener, url) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            return url
        body = response.read().decode("utf-8", errors="ignore")

    if "download-form" not in body:
        return url

    action_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', body)
    if not action_match:
        return url
    action = html.unescape(action_match.group(1))
    inputs = {
        html.unescape(name): html.unescape(value)
        for name, value in re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', body)
    }
    query = urllib.parse.urlencode(inputs)
    return f"{action}?{query}"


def download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".download")

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    final_url = resolve_google_drive_warning(url, opener)

    with request_url(opener, final_url) as response, temp_path.open("wb") as output_file:
        total_size = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = min(downloaded, total_size) * 100 / total_size
                sys.stdout.write(f"\rDownloading model: {percent:5.1f}%")
                sys.stdout.flush()

    sys.stdout.write("\n")
    if temp_path.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Downloaded model file is too small: {temp_path.stat().st_size} bytes")
    temp_path.replace(destination)


def ensure_model(model_key: str, model_dir: Path) -> tuple[Path, dict]:
    spec = MODEL_SPECS[model_key]
    model_path = model_dir / f"{spec['name']}.pth"
    if not model_path.exists():
        print(f"Model not found, downloading {spec['name']}...")
        urls = [spec["url"], *spec.get("fallback_urls", [])]
        last_error = None
        for url in urls:
            try:
                download_with_progress(url, model_path)
                break
            except Exception as exc:
                last_error = exc
                if model_path.with_suffix(model_path.suffix + ".download").exists():
                    model_path.with_suffix(model_path.suffix + ".download").unlink()
                print(f"Download failed from {url}: {exc}")
        else:
            raise RuntimeError(f"Could not download {spec['name']}.") from last_error
    return model_path, spec


def load_model(model_key: str, model_dir: Path, device: str, precision: str, channels_last: bool):
    torch = import_torch()
    model_path, spec = ensure_model(model_key, model_dir)

    if spec.get("loader") == "spandrel":
        ModelLoader, ImageModelDescriptor = import_spandrel()
        descriptor = ModelLoader(device="cpu").load_from_file(model_path)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise RuntimeError(f"Unsupported image model descriptor: {model_path}")
        if descriptor.scale != spec["scale"]:
            raise RuntimeError(f"Expected x{spec['scale']} model, got x{descriptor.scale}: {model_path}")

        if precision == "fp16":
            if not descriptor.supports_half:
                precision = "bf16" if descriptor.supports_bfloat16 and device == "cuda" else "fp32"
        if precision == "bf16" and not descriptor.supports_bfloat16:
            precision = "fp32"

        descriptor = descriptor.to(device).eval()
        if precision == "fp16":
            descriptor = descriptor.half()
        elif precision == "bf16":
            descriptor = descriptor.bfloat16()

        return descriptor, spec, precision

    model = build_rrdbnet(num_block=spec["num_block"], scale=spec["scale"])

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in checkpoint:
                state = checkpoint[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {model_path}")

    state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Model weights do not match {spec['name']}.\n"
            f"Missing keys: {missing[:8]}\nUnexpected keys: {unexpected[:8]}"
        )

    model.eval().to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.bfloat16()

    return model, spec, precision


def image_to_tensor(image: Image.Image, device: str, precision: str, channels_last: bool):
    torch = import_torch()
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    if precision == "fp16":
        tensor = tensor.half()
    elif precision == "bf16":
        tensor = tensor.bfloat16()
    return tensor


def tensor_to_image(tensor) -> Image.Image:
    torch = import_torch()
    tensor = tensor.detach().float().clamp_(0, 1).squeeze(0).permute(1, 2, 0).cpu()
    arr = (tensor.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def run_model(model, tile_tensor, tta: bool):
    torch = import_torch()
    if not tta:
        return model(tile_tensor)

    outputs = [model(tile_tensor)]

    hflip_in = torch.flip(tile_tensor, dims=[3])
    outputs.append(torch.flip(model(hflip_in), dims=[3]))

    vflip_in = torch.flip(tile_tensor, dims=[2])
    outputs.append(torch.flip(model(vflip_in), dims=[2]))

    hvflip_in = torch.flip(tile_tensor, dims=[2, 3])
    outputs.append(torch.flip(model(hvflip_in), dims=[2, 3]))

    return torch.stack(outputs, dim=0).mean(dim=0)


def emit_progress(current: int, total: int, started_at: float) -> None:
    elapsed = max(0.0, time.perf_counter() - started_at)
    eta = 0.0
    if current > 0 and total > current:
        eta = elapsed / current * (total - current)
    print(f"__PROGRESS__|{current}|{total}|{elapsed:.3f}|{eta:.3f}", flush=True)


def upscale_tensor_tiled(
    model,
    tensor,
    scale: int,
    tile: int,
    tile_pad: int,
    tta: bool,
    quiet: bool,
):
    torch = import_torch()
    _, channels, height, width = tensor.shape

    if tile <= 0:
        started_at = time.perf_counter()
        with torch.inference_mode():
            result = run_model(model, tensor, tta)
        emit_progress(1, 1, started_at)
        return result

    output = torch.empty(
        (1, channels, height * scale, width * scale),
        device=tensor.device,
        dtype=tensor.dtype,
        memory_format=torch.channels_last if tensor.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format,
    )

    x_tiles = math.ceil(width / tile)
    y_tiles = math.ceil(height / tile)
    total_tiles = x_tiles * y_tiles
    tile_index = 0
    started_at = time.perf_counter()

    with torch.inference_mode():
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                tile_index += 1
                x_end = min(x + tile, width)
                y_end = min(y + tile, height)

                x_pad_start = max(x - tile_pad, 0)
                y_pad_start = max(y - tile_pad, 0)
                x_pad_end = min(x_end + tile_pad, width)
                y_pad_end = min(y_end + tile_pad, height)

                if not quiet:
                    print(f"Tile {tile_index}/{total_tiles}: input {x}:{x_end}, {y}:{y_end}")

                tile_tensor = tensor[:, :, y_pad_start:y_pad_end, x_pad_start:x_pad_end]
                tile_output = run_model(model, tile_tensor, tta)

                crop_x = (x - x_pad_start) * scale
                crop_y = (y - y_pad_start) * scale
                crop_w = (x_end - x) * scale
                crop_h = (y_end - y) * scale

                output[:, :, y * scale : y_end * scale, x * scale : x_end * scale] = tile_output[
                    :, :, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w
                ]
                emit_progress(tile_index, total_tiles, started_at)

    return output


def upscale_with_oom_backoff(
    model,
    tensor,
    scale: int,
    tile: int,
    tile_pad: int,
    tta: bool,
    quiet: bool,
    auto_tile: bool,
):
    torch = import_torch()
    current_tile = tile
    while True:
        try:
            return upscale_tensor_tiled(model, tensor, scale, current_tile, tile_pad, tta, quiet), current_tile
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or not auto_tile or current_tile <= 48:
                raise
            torch.cuda.empty_cache()
            current_tile = max(48, current_tile // 2)
            print(f"CUDA memory was tight; retrying with tile={current_tile}.")


def read_image(path: Path) -> tuple[Image.Image, Image.Image | None]:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or ("transparency" in image.info):
        rgba = image.convert("RGBA")
        return rgba.convert("RGB"), rgba.getchannel("A")
    return image.convert("RGB"), None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_auto_float(value: str | float, low: float, high: float, name: str) -> float | str:
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be 'auto' or a number.") from exc
    return clamp(parsed, low, high)


def estimate_image_characteristics(image: Image.Image) -> dict[str, float]:
    sample = image.convert("L")
    if max(sample.size) > 640:
        ratio = 640 / max(sample.size)
        sample = sample.resize(
            (max(1, round(sample.width * ratio)), max(1, round(sample.height * ratio))),
            Image.Resampling.LANCZOS,
        )

    gray = np.asarray(sample).astype(np.float32) / 255.0
    if min(gray.shape) < 4:
        return {"noise": 0.0, "edge": 0.0, "texture": 0.0}

    median = np.asarray(sample.filter(ImageFilter.MedianFilter(size=3))).astype(np.float32) / 255.0
    noise = float(np.mean(np.abs(gray - median)))

    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * center
    )
    edge = float(np.mean(np.abs(laplacian)))
    texture = float(np.std(center))
    return {"noise": noise, "edge": edge, "texture": texture}


def automatic_postprocess_values(image: Image.Image, sharpness: str | float, denoise: str | float) -> tuple[float, float]:
    characteristics = estimate_image_characteristics(image)
    noise = characteristics["noise"]
    edge = characteristics["edge"]
    texture = characteristics["texture"]

    if denoise == "auto":
        denoise_value = clamp(0.025 + noise * 3.1 + max(0.0, texture - 0.22) * 0.12, 0.0, 0.24)
    else:
        denoise_value = float(denoise)

    if sharpness == "auto":
        sharpness_value = 0.92 - edge * 5.2 - denoise_value * 0.9
        if texture > 0.28:
            sharpness_value -= 0.10
        if noise < 0.012 and edge < 0.035:
            sharpness_value += 0.12
        sharpness_value = clamp(sharpness_value, 0.30, 1.10)
    else:
        sharpness_value = float(sharpness)

    return sharpness_value, denoise_value


def postprocess(
    image: Image.Image,
    source_image: Image.Image,
    sharpness: str | float,
    denoise: str | float,
    contrast: float,
    quiet: bool,
) -> Image.Image:
    output = image
    sharpness_value, denoise_value = automatic_postprocess_values(source_image, sharpness, denoise)

    if not quiet:
        print(f"Auto postprocess: sharpness={sharpness_value:.2f} | denoise={denoise_value:.2f}")

    if denoise_value > 0:
        smoothed = output.filter(ImageFilter.MedianFilter(size=3))
        output = Image.blend(output, smoothed, clamp(denoise_value, 0.0, 1.0) * 0.35)

    if sharpness_value > 0:
        percent = int(45 + 120 * clamp(sharpness_value, 0.0, 2.0))
        radius = 0.9 if sharpness_value <= 1.0 else 1.15
        output = output.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=2))
        output = ImageEnhance.Sharpness(output).enhance(1.0 + sharpness_value * 0.18)

    if contrast != 1.0:
        output = ImageEnhance.Contrast(output).enhance(contrast)

    return output


def resize_alpha(alpha: Image.Image | None, size: tuple[int, int]) -> Image.Image | None:
    if alpha is None:
        return None
    return alpha.resize(size, Image.Resampling.LANCZOS)


def choose_output_extension(input_path: Path, output_format: str, alpha: bool) -> str:
    if output_format == "auto":
        return ".png" if alpha else input_path.suffix
    if output_format == "jpeg" and alpha:
        return ".png"
    return OUTPUT_FORMAT_EXTENSIONS[output_format] or input_path.suffix


def make_output_path(input_path: Path, output_arg: Path | None, suffix: str, alpha: bool, output_format: str) -> Path:
    ext = choose_output_extension(input_path, output_format, alpha)
    if output_arg is None:
        return input_path.with_name(f"{input_path.stem}{suffix}{ext}")

    if output_arg.suffix:
        if output_format != "auto":
            return output_arg.with_suffix(ext)
        if alpha and output_arg.suffix.lower() in {".jpg", ".jpeg"}:
            return output_arg.with_suffix(".png")
        return output_arg

    output_arg.mkdir(parents=True, exist_ok=True)
    return output_arg / f"{input_path.stem}{suffix}{ext}"


def discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")
    files = sorted(path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"No supported images found in: {input_path}")
    return files


def upscale_one_image(args, model, spec: dict, input_path: Path, output_path: Path) -> None:
    torch = import_torch()
    rgb, alpha = read_image(input_path)
    print(f"Input: {input_path} ({rgb.width}x{rgb.height})")

    tensor = image_to_tensor(rgb, args.device, args.precision, args.channels_last)
    start = time.perf_counter()

    output_tensor, used_tile = upscale_with_oom_backoff(
        model=model,
        tensor=tensor,
        scale=spec["scale"],
        tile=args.tile,
        tile_pad=args.tile_pad,
        tta=args.tta,
        quiet=args.quiet,
        auto_tile=args.auto_tile,
    )

    output = tensor_to_image(output_tensor)

    if args.outscale != spec["scale"]:
        target_size = (
            max(1, round(rgb.width * args.outscale)),
            max(1, round(rgb.height * args.outscale)),
        )
        output = output.resize(target_size, Image.Resampling.LANCZOS)

    output = postprocess(
        output,
        source_image=rgb,
        sharpness=args.sharpness,
        denoise=args.denoise,
        contrast=args.contrast,
        quiet=args.quiet,
    )
    alpha_out = resize_alpha(alpha, output.size)
    if alpha_out is not None:
        output = Image.merge("RGBA", (*output.split(), alpha_out))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs.update({"quality": args.jpeg_quality, "subsampling": 0})
    elif output_path.suffix.lower() == ".webp":
        save_kwargs.update({"quality": args.jpeg_quality, "method": 6})
    elif output_path.suffix.lower() == ".png":
        save_kwargs.update({"compress_level": 4})

    output.save(output_path, **save_kwargs)
    torch.cuda.synchronize() if args.device == "cuda" else None
    elapsed = time.perf_counter() - start
    print(f"Output: {output_path} ({output.width}x{output.height}) | tile={used_tile} | {elapsed:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUDA DLSS-like image super-resolution using Real-ESRGAN, HAT, and ATD weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input image file or folder.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output file or folder.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Folder used to cache model weights.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="atd", help="Model preset.")
    parser.add_argument("--outscale", type=float, default=4.0, help="Final output scale. The model runs at x4 first.")
    parser.add_argument("--suffix", default="_sr", help="Output filename suffix when output is a folder or omitted.")
    parser.add_argument("--output-format", choices=sorted(OUTPUT_FORMAT_EXTENSIONS), default="auto", help="Output image format.")

    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Inference device. CUDA is expected.")
    parser.add_argument("--gpu-usage", choices=["auto", "conservative", "balanced", "performance"], default="auto", help="GPU memory and speed profile.")
    parser.add_argument("--gpu-memory-fraction", default="auto", help="Maximum CUDA memory fraction, or auto.")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto", help="Inference precision.")
    parser.add_argument("--tile", default="auto", help="Input tile size, or auto.")
    parser.add_argument("--tile-pad", default="auto", help="Extra pixels around each tile, or auto.")
    parser.add_argument("--no-auto-tile", dest="auto_tile", action="store_false", help="Disable automatic tile-size retry on OOM.")
    parser.add_argument("--channels-last", choices=["auto", "on", "off"], default="auto", help="Use channels-last CUDA layout.")
    parser.add_argument("--tta", action="store_true", help="Use flip test-time augmentation. Better edges, much slower.")

    parser.add_argument("--sharpness", default="auto", help="DLSS-style post-sharpening strength, or auto.")
    parser.add_argument("--denoise", default="auto", help="Light denoise before sharpening, or auto.")
    parser.add_argument("--contrast", type=float, default=1.0, help="Final contrast multiplier.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG output quality.")
    parser.add_argument("--quiet", action="store_true", help="Hide per-tile progress.")

    parser.set_defaults(auto_tile=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.outscale <= 0:
        raise SystemExit("--outscale must be greater than 0.")
    if args.tile != "auto" and int(args.tile) < 0:
        raise SystemExit("--tile must be 'auto' or 0 or greater.")
    if args.tile_pad != "auto" and int(args.tile_pad) < 0:
        raise SystemExit("--tile-pad must be 'auto' or 0 or greater.")
    args.jpeg_quality = max(1, min(100, args.jpeg_quality))

    args.sharpness = parse_auto_float(args.sharpness, 0.0, 2.0, "--sharpness")
    args.denoise = parse_auto_float(args.denoise, 0.0, 1.0, "--denoise")
    spec = MODEL_SPECS[args.model]

    torch = import_torch()
    profile = cuda_profile(args.device)
    if args.device == "cuda":
        print(
            f"GPU: {profile.name} | capability {profile.capability} | "
            f"VRAM {profile.vram_gb:.1f} GB | free {profile.free_gb:.1f} GB | SM {profile.multiprocessors}"
        )
    apply_runtime_tuning(args, spec, profile)

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model, spec, actual_precision = load_model(
        model_key=args.model,
        model_dir=args.model_dir,
        device=args.device,
        precision=args.precision,
        channels_last=args.channels_last,
    )
    args.precision = actual_precision

    input_paths = discover_inputs(args.input)
    output_arg = args.output
    if args.input.is_dir() and output_arg is not None and output_arg.suffix:
        raise SystemExit("When input is a folder, --output must also be a folder.")
    if args.input.is_dir() and output_arg is None:
        output_arg = args.input / "upscaled"

    for input_path in input_paths:
        rgb, alpha = read_image(input_path)
        rgb.close()
        if alpha is not None:
            alpha.close()
        output_path = make_output_path(input_path, output_arg, args.suffix, alpha is not None, args.output_format)
        upscale_one_image(args, model, spec, input_path, output_path)


if __name__ == "__main__":
    main()
