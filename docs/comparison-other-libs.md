# Comparison with other GPU-based data transforms library

## 📊 Triton-Augment vs DALI vs Kornia

| Feature                      | **Triton-Augment** (yours)                                                                  | **NVIDIA DALI**                                                                      | **Kornia**                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------- |
| **Primary goal**             | Fast, fused GPU augmentations for training                                                  | End-to-end input pipeline (decode → resize → augment)                                | Differentiable augmentations in pure PyTorch        |
| **Fused ops**                | ✔️ Crop + flip + brightness + contrast + saturation + grayscale + normalize (single kernel) | ⚠️ Only some fusions (e.g., `crop_mirror_normalize`); color ops are separate kernels | ❌ No fusion — each op is a separate CUDA/PyTorch op |
| **Per-sample random params** | ✔️ Built-in, torchvision-style API                                                          | ✔️ Supported (via feeding random tensors), but more manual                           | ✔️ Built-in                                         |
| **Ease of use**              | ✔️ Simple, torchvision-like                                                                 | ⚠️ Steeper learning curve (pipeline graph)                                           | ✔️ Very easy (just PyTorch ops)                     |
| **Supported ops**            | ⚠️ Limited for now (crop, flip, color jitter, normalize, grayscale)                         | ✔️ Huge library (decode, resize, warp, color, video, audio)                          | ✔️ Wide set (geometry, color, filtering, keypoints) |
| **Performance**              | 🚀 Very fast for augmentation (1 fused kernel for all ops)                                  | 🚀 Fast for full pipelines (GPU decode/resize), but augmentation uses multiple kernels (less fusion) | ⚠️ Moderate (PyTorch kernels, multiple launches)    |
| **Integration**              | PyTorch training pipelines                                                                  | PyTorch, TensorFlow, JAX                                                             | PyTorch only                                        |
| **CPU preprocessing**        | ❌ None (expects tensors already on GPU)                                                     | ✔️ Hardware-accelerated decode/resize possible                                       | ✔️ Built on top of PyTorch                                          |
| **Autograd support**         | ❌ Not needed (augmentations only)                                                           | ❌ Most ops are not differentiable                                                    | ✔️ Yes (Kornia is differentiable by design)         |
| **Production readiness**     | ⚠️ Early-stage (fast but limited scope)                                                     | ✔️ Mature, used in industry                                                          | ✔️ Mature                                           |

---

## 📝 Notes

* **Triton-Augment is not a replacement for DALI or Kornia.**
  It’s a *small, focused* library aimed at speeding up a few high-impact augmentations via kernel fusion.

* **DALI is still the best choice** if the bottleneck is decode/resize or you need full data pipeline acceleration. However, for augmentation-only workloads (data already on GPU as tensors), Triton-Augment is faster due to higher kernel fusion.

* **Kornia is best** if you need differentiable augmentation or a wide variety of transforms.

* **Our advantage**:
  For the operations that are supported, **our one-kernel design beats both** in raw speed and simplicity.

* **Our limitation**:
  Fewer ops, no CPU pipeline, not designed for everything — just the common path.

