# cudaembedr 0.2.0

- Embedding stages now use the shared validated provenance schema, including
  requested device, actual device, backend, fallback reason, and output device.
- Embeddings retain the full provenance and aggregate compute device of
  compatible cudaverse inputs.
- Diffusion maps preserve the original device request and continue to report
  CUDA distance plus CPU kernel/eigendecomposition as hybrid execution.
- UMAP, t-SNE, and diffusion results now retain their complete normalized
  explicit parameters.

# cudaembedr 0.1.2

- Added an end-to-end single-cell workflow test proving that cell identifiers
  remain attached to embedding coordinates.

# cudaembedr 0.1.1

- Embedding results now expose stable per-stage `compute_stages` provenance.
- Diffusion maps correctly report `compute_device = "hybrid"` when CUDA
  distance calculation is followed by CPU kernel construction and
  eigendecomposition.
- Corrected the t-SNE `...` documentation to reference `Rtsne::Rtsne()`.
