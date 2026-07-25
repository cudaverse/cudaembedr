# cudaembedr 0.1.1

- Embedding results now expose stable per-stage `compute_stages` provenance.
- Diffusion maps correctly report `compute_device = "hybrid"` when CUDA
  distance calculation is followed by CPU kernel construction and
  eigendecomposition.
- Corrected the t-SNE `...` documentation to reference `Rtsne::Rtsne()`.
