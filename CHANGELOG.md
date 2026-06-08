# Changelog

## v1.1.3 - 2026-06-08

Runtime installation fix release.

- Runtime installation now uses the current user's local app data folder to avoid write failures on the desktop, Program Files, or other protected install locations.
- If the Python installer fails, the app downloads a fresh installer and retries.
- Python installation now shows the installer log path for easier diagnosis on another PC.
- If the official Python installer installs the same Python version into its default directory, the app detects it and continues creating the runtime environment.
- The launcher script supports the new runtime folder while keeping compatibility with the old `.venv` folder.

## v1.1.2 - 2026-06-08

Single-file installer release.

- Release asset changed to `ImageSuperResolutionTool-EN-v1.1.2-Setup.exe`.
- Installer supports choosing the install location.
- Desktop shortcut is created automatically after installation.
- Installer includes four models: ATD, HAT, Real-ESRGAN general photo, and Real-ESRGAN anime/illustration.
- The app window title continues to show the current version at the end.
- Model dropdown order: ATD, HAT, Real-ESRGAN general photo, Real-ESRGAN anime/illustration.
- Fixes startup on PCs without system Python: the desktop shortcut opens the app UI instead of a Python website.
- The app can open first, then `Install/Check Runtime` installs the local Python runtime and CUDA dependencies.
- Old zip downloaders, standalone runtime scripts, and old launch entries are not included.
