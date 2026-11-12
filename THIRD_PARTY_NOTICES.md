# Third-party notices

`src/nctd/vendor/ddpm_torch/` contains three unmodified source files and minimal
package initializers. These files were obtained from
[ku-dmlab/Adaptive-Timestep-Sampler](https://github.com/ku-dmlab/Adaptive-Timestep-Sampler),
whose README credits [tqch/ddpm-torch](https://github.com/tqch/ddpm-torch).

- `functions.py`
- `modules.py`
- `models/unet.py`

The applicable MIT copyright and license texts are included in
[licenses/Adaptive-Timestep-Sampler-MIT.txt](licenses/Adaptive-Timestep-Sampler-MIT.txt)
(Copyright 2025 ku-dmlab) and
[licenses/ddpm-torch-MIT.txt](licenses/ddpm-torch-MIT.txt)
(Copyright 2022 Tianqi Chen).

The U-Net implementation is third-party work. The experiment adaptations are
the noise-conditioned convolution wrapper, local trajectory training setup,
controls, and the portable orchestration described in the documentation.
Standard VP diffusion, v prediction, DDIM and teacher-guided training are not
presented as new foundational algorithms.

Original local research source filenames and SHA256 hashes are recorded in
[docs/source_provenance.json](docs/source_provenance.json). They describe the
origin of this release; no original workspace path is required at runtime.

This release does not redistribute CIFAR-10, pretrained Inception weights,
historical model checkpoints or the Inception implementation. PyTorch, NumPy
and Pillow are installed as separate dependencies under their own licenses.

No blanket license has been assigned to the original project portions in this
archive. Public availability does not extend the third-party MIT license to all
other files.
