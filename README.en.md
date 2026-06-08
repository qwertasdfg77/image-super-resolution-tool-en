# Image Super Resolution Tool

A Windows GUI image upscaling tool. Current public version: `v1.1.3`.

## Download

Download the single installer from the Latest Release page:

- `ImageSuperResolutionTool-EN-v1.1.3-Setup.exe`

Latest Release:
https://github.com/qwertasdfg77/image-super-resolution-tool-en/releases/latest

## Usage

1. Run `ImageSuperResolutionTool-EN-v1.1.3-Setup.exe`.
2. Choose an install location.
3. Open the desktop shortcut created by the installer.
4. On first use, click `Install/Check Runtime` in the app.
5. Select an input image or folder, select an output folder, then start upscaling.

The installer includes four models: ATD, HAT, Real-ESRGAN photo, and Real-ESRGAN anime/illustration. The app UI opens even if Python is not installed system-wide; the in-app `Install/Check Runtime` button installs the local Python runtime, CUDA PyTorch, and other dependencies.

Recommended GPU: NVIDIA RTX 4060 8GB or higher.
