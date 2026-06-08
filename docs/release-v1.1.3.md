## Image Super Resolution Tool v1.1.3 Runtime Installation Fix

Release file:

`ImageSuperResolutionTool-EN-v1.1.3-Setup.exe`

## Fixes

- Fixes the case where `Install/Check Runtime` finishes the Python installer but `.python\python.exe` is missing on another PC.
- The local Python runtime and `.venv` are now stored under the current user's local app data folder.
- If the Python installer fails, the app downloads a fresh installer and retries.
- The installer window shows the Python installer log path for easier troubleshooting.
- The launcher script supports the new runtime folder while keeping compatibility with the old `.venv` folder.

## Usage

Download and run `ImageSuperResolutionTool-EN-v1.1.3-Setup.exe`, then choose an install location. After installation, open the app from the desktop shortcut.

On first launch, click `Install/Check Runtime` in the app to install the local Python runtime, CUDA PyTorch, and other dependencies.
