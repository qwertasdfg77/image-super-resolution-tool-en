# Image Super Resolution Tool

A Windows GUI image upscaling tool. Current public version: `v1.1.2`. It is designed for regular users and does not require writing code.

## Download

Download one installer from the Latest Release page:

- `ImageSuperResolutionTool-EN-v1.1.2-Setup.exe`

Latest Release:
https://github.com/qwertasdfg77/image-super-resolution-tool-en/releases/latest

## Usage

1. Download `ImageSuperResolutionTool-EN-v1.1.2-Setup.exe`.
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

The installer includes a launcher that opens the app UI directly. If Python is not installed system-wide, the app should still open first; then `Install/Check Runtime` downloads the local Python runtime, CUDA PyTorch, and other dependencies.

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
