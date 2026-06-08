# Changelog

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
