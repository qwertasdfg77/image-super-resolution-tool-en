# Third-Party Components and Model Notes

This project uses third-party open-source libraries, model architectures, and pretrained model weights. The source code is MIT licensed; third-party components and models remain governed by their upstream licenses or terms.

## Python Dependencies

| Name | Purpose | Upstream | License Information |
| --- | --- | --- | --- |
| PyTorch | CUDA inference and tensor computation | https://github.com/pytorch/pytorch | See upstream PyTorch LICENSE |
| torchvision | PyTorch vision dependency | https://github.com/pytorch/vision | BSD-3-Clause |
| Spandrel | Loading ATD and HAT Transformer models | https://github.com/chaiNNer-org/spandrel | MIT |
| safetensors | Safe tensor weight reading dependency used by Spandrel | https://github.com/huggingface/safetensors | Apache-2.0 |
| Pillow | Image loading, saving, and post-processing | https://github.com/python-pillow/Pillow | See upstream Pillow LICENSE |
| NumPy | Array and image data processing | https://github.com/numpy/numpy | See upstream NumPy LICENSE |
| packaging | Python package version handling | https://github.com/pypa/packaging | See upstream packaging LICENSE |

## Models and Algorithms

| Name | Purpose | Upstream | License Information |
| --- | --- | --- | --- |
| ATD | Official super-resolution model | https://github.com/LabShuHangGU/Adaptive-Token-Dictionary | Apache-2.0 |
| HAT | High-quality super-resolution model | https://github.com/XPixelGroup/HAT | Apache-2.0 |
| Real-ESRGAN | General photo and anime/illustration upscaling models | https://github.com/xinntao/Real-ESRGAN | BSD-3-Clause |

## Bundled Models

The release installer includes these model files:

- `003_ATD_SRx4_finetune.pth`
- `Real_HAT_GAN_sharper.pth`
- `RealESRGAN_x4plus.pth`
- `RealESRGAN_x4plus_anime_6B.pth`

ATD weights come from the official ATD pretrained model. OpenModelDB lists SHA256 `092bf6aa82f0ecb9f681a7da6a8e65e2c1280ae5b44f1608f60742d88547b833`. HAT weights come from the public mirror repository `Acly/hat`. If redistributing this project, keep this file, `LICENSE`, upstream license links, and model source notes.

This project is not NVIDIA DLSS. DLSS is an in-game multi-frame real-time upsampling technology; this project processes still images or image folders.
