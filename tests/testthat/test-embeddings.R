embedding_data <- function(n = 40L) {
  set.seed(1)
  x <- matrix(rnorm(n * 4L), n, 4L)
  rownames(x) <- paste0("cell", seq_len(n))
  x
}

test_that("diffusion embedding records dimensions and backend", {
  fit <- cuda_diffusion_map(
    embedding_data(),
    n_components = 3,
    device = "cpu"
  )

  expect_s3_class(fit, "cuda_embedding")
  expect_identical(dim(fit$coordinates), c(40L, 3L))
  expect_identical(rownames(fit$coordinates), paste0("cell", 1:40))
  expect_identical(fit$compute_device, "cpu")
  expect_length(fit$eigenvalues, 3)
  expect_equal(embedding_coordinates(fit), fit$coordinates)
})

test_that("cuda_pca inputs compose with embeddings", {
  pca <- cudalearnr::cuda_pca(
    embedding_data(),
    n_components = 3,
    device = "cpu"
  )
  fit <- cuda_diffusion_map(pca, n_components = 2, device = "cpu")

  expect_identical(fit$source_class, "cuda_pca")
  expect_identical(fit$source_device, "cpu")
  expect_identical(dim(fit$coordinates), c(40L, 2L))
})

test_that("t-SNE adapter returns a common result", {
  skip_if_not_installed("Rtsne")
  fit <- cuda_tsne(
    embedding_data(),
    perplexity = 5,
    seed = 1,
    max_iter = 250
  )

  expect_s3_class(fit, "cuda_embedding")
  expect_identical(dim(fit$coordinates), c(40L, 2L))
  expect_identical(fit$backend, "Rtsne")
})

test_that("UMAP adapter returns a common result", {
  skip_if_not_installed("uwot")
  fit <- cuda_umap(
    embedding_data(),
    n_neighbors = 5,
    n_epochs = 20,
    seed = 1
  )

  expect_s3_class(fit, "cuda_embedding")
  expect_identical(dim(fit$coordinates), c(40L, 2L))
  expect_identical(fit$backend, "uwot")
})

test_that("invalid embedding parameters fail clearly", {
  expect_error(
    cuda_diffusion_map(embedding_data(), n_components = 100),
    "whole number"
  )
  expect_error(
    cuda_diffusion_map(embedding_data(), sigma = 0),
    "positive finite"
  )
  expect_error(
    cuda_tsne(embedding_data(), perplexity = 20),
    "perplexity"
  )
  expect_error(embedding_coordinates(matrix(1)), "cuda_embedding")
})
