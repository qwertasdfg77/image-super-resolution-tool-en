# Security Notes

## Supported Versions

| Version | Status |
| --- | --- |
| v1.1.2 | Supported |

## Download Safety

- Download the single-file installer only from this project release page or another trusted source from the author.
- The single-file installer includes ATD Official Super-Resolution Model, HAT High-Quality Super-Resolution Model, Real-ESRGAN General Photo Model, and Real-ESRGAN Anime and Illustration Model, but it does not pre-bundle `.venv`.
- On first use, the in-app `Install/Check Runtime` window downloads the local Python runtime, CUDA PyTorch, and other dependencies.
- If Python is not installed system-wide, the app should still open first; users do not need to manually visit python.org.
- Avoid repackaged installers from third-party file hosts or unknown sources.

## Reporting Security Issues

If you find a security issue, do not post exploitable details in a public Issue. Create an Issue without sensitive details and state that private coordination is needed.
