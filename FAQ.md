# FAQ

## Why is the installer large?

Because the `v1.1.3` installer includes four super-resolution models, so users do not need separate model downloads.

## Do I still need to install a runtime?

Yes. The installer includes models and a launcher that opens the app UI directly, but it does not pre-bundle `.venv`. On first use, click `Install/Check Runtime` to install the local Python runtime, CUDA PyTorch, and other dependencies.

## How do I use the one-click installer?

Download `ImageSuperResolutionTool-EN-v1.1.3-Setup.exe` from Release, run it, and choose an install location. A desktop shortcut will be created after installation.

## Will my photos be uploaded to GitHub?

No. The repository and release package do not contain user photos. The app also does not save the previous input image path or output folder path.

## What GPU is recommended?

NVIDIA RTX 4060 8GB or higher is recommended. The app automatically detects GPU model, VRAM, and free VRAM, then adjusts processing usage.

## Do I need to download models separately?

No. The installer already includes ATD, HAT, Real-ESRGAN general photo, and Real-ESRGAN anime/illustration models.
