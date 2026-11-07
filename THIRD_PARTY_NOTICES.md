# Third-party notices

FallGuard Lite is licensed under the GNU Affero General Public License v3.0 or
later. It depends on the following third-party projects. Their own licenses and
copyright notices continue to apply.

| Component | Version | Purpose | License |
| --- | ---: | --- | --- |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | 8.4.202 | YOLO pose inference | AGPL-3.0 |
| `yolo11n-pose.pt` | YOLO11 | Pretrained person pose weights | Ultralytics AGPL-3.0 / Enterprise terms |
| [PyTorch](https://github.com/pytorch/pytorch) | 2.14.1 | Tensor inference runtime | BSD-style |
| [TorchVision](https://github.com/pytorch/vision) | 0.29.1 | Vision runtime support | BSD-3-Clause |
| [OpenCV](https://github.com/opencv/opencv-python) | 4.14.0.94 | JPEG decoding | Apache-2.0 |
| [FastAPI](https://github.com/fastapi/fastapi) | 0.162.0 | Web application framework | MIT |
| [Uvicorn](https://github.com/Kludex/uvicorn) | 0.54.0 | ASGI server | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | 2.3.5 | Array processing | BSD-3-Clause |

Ultralytics offers AGPL-3.0 and Enterprise licensing options. Review its current
licensing terms before commercial, proprietary, or hosted production use:
<https://www.ultralytics.com/license>.

No third-party sample video, face image, or training dataset is redistributed by
this repository. The Docker build obtains the named pretrained model through the
Ultralytics package, verifies SHA-256
`869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`, and
stores that digest in a sidecar that is checked again at runtime.
