# Image Super Resolution Tool

[![Latest Release](https://img.shields.io/github/v/release/qwertasdfg77/image-super-resolution-tool-en?label=latest)](https://github.com/qwertasdfg77/image-super-resolution-tool-en/releases/latest)
[![Download](https://img.shields.io/badge/download-single%20installer-brightgreen)](https://github.com/qwertasdfg77/image-super-resolution-tool-en/releases/latest)
[![Windows](https://img.shields.io/badge/Windows-10%20%2F%2011-0078d4)](https://github.com/qwertasdfg77/image-super-resolution-tool-en/releases/latest)
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76b900)](https://developer.nvidia.com/cuda-zone)
[![Models](https://img.shields.io/badge/models-ATD%20%7C%20HAT%20%7C%20Real--ESRGAN-orange)](THIRD_PARTY_NOTICES.md)
[![License](https://img.shields.io/github/license/qwertasdfg77/image-super-resolution-tool-en)](LICENSE)

A Windows GUI image upscaling tool. Current public version: `v1.1.3`. It is designed for regular users and does not require writing code.

Keywords: image super-resolution, AI upscaling, photo enhancement, anime upscaling, NVIDIA CUDA, RTX 4060, ATD, HAT, Real-ESRGAN.

Chinese version: [Chinese repository](https://github.com/qwertasdfg77/image-super-resolution-tool)

## Download

Download one installer from the Latest Release page:

- `ImageSuperResolutionTool-EN-v1.1.3-Setup.exe`

Latest Release:
https://github.com/qwertasdfg77/image-super-resolution-tool-en/releases/latest

## Usage

1. Download `ImageSuperResolutionTool-EN-v1.1.3-Setup.exe`.
2. Run the installer.
3. Choose an install location.
4. After installation, the desktop shortcut `Image Super Resolution Tool` will be created.
5. On first launch, click `Install/Check Runtime` in the app.

## What The Installer Includes

- The graphical app and super-resolution engine script.
- ATD Official Super-Resolution Model.
- HAT High-Quality Super-Resolution Model.
- Real-ESRGAN General Photo Model.
- Real-ESRGAN Anime and Illustration Model.
- App icon and desktop shortcut creation logic.

The installer includes a launcher that opens the app UI directly. If Python is not installed system-wide, the app should still open first; then `Install/Check Runtime` downloads the local Python runtime, CUDA PyTorch, and other dependencies. The runtime is installed under the current user's local app data folder to avoid write failures when the app is installed on the desktop or in a protected folder.

## Main Features

- Automatically detects NVIDIA GPU model, VRAM capacity, and free VRAM.
- Automatically selects suitable processing usage for the detected GPU.
- Uses `ATD Official Super-Resolution Model` by default.
- Includes four models, so users do not need separate model downloads.
- Automatically adjusts sharpening and denoising.
- Shows a percentage progress bar with elapsed time and estimated remaining time.
- Supports single-image and folder batch processing.
- Does not save the previous input image path or output folder path.
- Does not show before/after preview panels inside the app.

## Recommended Hardware

- Windows 10 / Windows 11
- NVIDIA RTX 4060 8GB or higher
- First runtime installation requires a stable network and enough disk space

## Notes

The source repository does not store large `.pth` model files directly. Complete models and the installer are published through GitHub Releases.

This tool is not native NVIDIA DLSS. Native DLSS in games requires engine data such as multiple frames, motion vectors, and depth information. This tool processes still images or image folders.
