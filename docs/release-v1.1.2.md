## Image Super Resolution Tool v1.1.2 Single-File Installer

This is the recommended public English release. Regular users only need to download one installer file:

`ImageSuperResolutionTool-EN-v1.1.2-Setup.exe`

### Changes

- Single-file installer release; no zip package or separate downloader script.
- Fixes startup on PCs without system Python: the desktop shortcut now opens the app UI instead of a Python website.
- Installer lets users choose the install location.
- Desktop shortcut is created automatically after installation.
- Installer includes four models:
  - ATD Official Super-Resolution Model
  - HAT High-Quality Super-Resolution Model
  - Real-ESRGAN General Photo Model
  - Real-ESRGAN Anime and Illustration Model
- Old downloaders, standalone runtime scripts, and old launch entries are not included.

### Usage

Download and run `ImageSuperResolutionTool-EN-v1.1.2-Setup.exe`, then choose an install location. After installation, open the app from the desktop shortcut.

On first launch, click `Install/Check Runtime` in the lower-left area to install the local Python runtime, CUDA PyTorch, and other dependencies.
