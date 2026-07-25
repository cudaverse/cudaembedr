.embedding_input <- function(x) {
  source_device <- "cpu"
  source_class <- class(x)[[1L]]
  if (inherits(x, "cudacell_workflow")) {
    source_device <- x$pca$device %||% "unknown"
    x <- x$pca$x
  } else if (inherits(x, "cuda_pca")) {
    source_device <- x$device %||% "unknown"
    x <- x$x
  } else if (inherits(x, "cudatensor")) {
    source_device <- x$device %||% "unknown"
    x <- as.array(x)
  } else if (is.data.frame(x)) {
    x <- as.matrix(x)
  }
  if (!is.matrix(x) || !is.numeric(x) || nrow(x) < 3L || ncol(x) < 1L ||
      anyNA(x) || any(!is.finite(x))) {
    stop(
      "`x` must resolve to a finite numeric matrix with at least three rows.",
      call. = FALSE
    )
  }
  list(
    matrix = unname(x),
    row_names = rownames(x),
    source_device = source_device,
    source_class = source_class
  )
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) y else x
}

.embedding_components <- function(n_components, maximum) {
  if (!is.numeric(n_components) || length(n_components) != 1L ||
      is.na(n_components) || !is.finite(n_components) ||
      n_components < 1 || n_components > maximum ||
      n_components != as.integer(n_components)) {
    stop(
      sprintf("`n_components` must be a whole number between 1 and %s.",
              maximum),
      call. = FALSE
    )
  }
  as.integer(n_components)
}

.embedding_seed <- function(seed) {
  if (is.null(seed)) {
    return(NULL)
  }
  if (!is.numeric(seed) || length(seed) != 1L || is.na(seed) ||
      !is.finite(seed)) {
    stop("`seed` must be NULL or one finite whole number.", call. = FALSE)
  }
  integer_seed <- suppressWarnings(as.integer(seed))
  if (is.na(integer_seed) || seed != integer_seed) {
    stop("`seed` must be NULL or one finite whole number.", call. = FALSE)
  }
  integer_seed
}

.with_embedding_seed <- function(seed, code) {
  seed <- .embedding_seed(seed)
  if (is.null(seed)) {
    return(force(code))
  }

  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) {
    old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  set.seed(seed)
  force(code)
}

.new_embedding <- function(coordinates, method, backend, input, parameters,
                           compute_device = "cpu", compute_stages = NULL) {
  if (is.null(compute_stages)) {
    compute_stages <- list(
      embedding = list(device = compute_device, backend = backend)
    )
  }
  if (!is.null(input$row_names)) {
    rownames(coordinates) <- input$row_names
  }
  colnames(coordinates) <- paste0(
    toupper(method),
    seq_len(ncol(coordinates))
  )
  structure(
    list(
      coordinates = coordinates,
      method = method,
      backend = backend,
      compute_device = compute_device,
      compute_stages = compute_stages,
      source_device = input$source_device,
      source_class = input$source_class,
      parameters = parameters
    ),
    class = "cuda_embedding"
  )
}

#' UMAP embedding
#'
#' UMAP currently uses the CPU `uwot` backend. GPU-aware cudaverse inputs are
#' accepted and their source device is retained in the result metadata.
#'
#' @param x Numeric observation-by-feature matrix or compatible cudaverse result.
#' @param n_components Output dimensions.
#' @param n_neighbors Number of nearest neighbours.
#' @param min_dist Minimum UMAP distance.
#' @param metric Distance metric passed to `uwot::umap()`.
#' @param n_epochs Optional training epochs.
#' @param seed Optional random seed.
#' @param ... Additional arguments passed to `uwot::umap()`.
#' @return A `cuda_embedding` list containing `coordinates`, `method`,
#'   `backend`, `compute_device`, per-stage `compute_stages`, source metadata,
#'   and algorithm `parameters`.
#' @export
#' @examples
#' if (requireNamespace("uwot", quietly = TRUE)) {
#'   cuda_umap(matrix(rnorm(120), 40, 3), n_neighbors = 5, seed = 1)
#' }
cuda_umap <- function(x, n_components = 2L, n_neighbors = 15L,
                      min_dist = 0.1, metric = "euclidean",
                      n_epochs = NULL, seed = NULL, ...) {
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("Install the 'uwot' package to compute UMAP embeddings.",
         call. = FALSE)
  }
  input <- .embedding_input(x)
  n_components <- .embedding_components(
    n_components,
    max(1L, nrow(input$matrix) - 2L)
  )
  if (!is.numeric(n_neighbors) || length(n_neighbors) != 1L ||
      is.na(n_neighbors) || n_neighbors < 2L ||
      n_neighbors >= nrow(input$matrix) ||
      n_neighbors != as.integer(n_neighbors)) {
    stop("`n_neighbors` must be between 2 and nrow(x) - 1.",
         call. = FALSE)
  }
  if (!is.numeric(min_dist) || length(min_dist) != 1L ||
      is.na(min_dist) || !is.finite(min_dist) || min_dist < 0) {
    stop("`min_dist` must be a finite non-negative number.",
         call. = FALSE)
  }
  if (!is.character(metric) || length(metric) != 1L || is.na(metric)) {
    stop("`metric` must be one character string.", call. = FALSE)
  }
  arguments <- list(
    X = input$matrix,
    n_neighbors = as.integer(n_neighbors),
    n_components = n_components,
    min_dist = min_dist,
    metric = metric,
    ret_model = FALSE,
    verbose = FALSE,
    ...
  )
  if (!is.null(n_epochs)) {
    arguments$n_epochs <- n_epochs
  }
  coordinates <- .with_embedding_seed(
    seed,
    do.call(uwot::umap, arguments)
  )
  .new_embedding(
    coordinates,
    method = "umap",
    backend = "uwot",
    input = input,
    parameters = list(
      n_neighbors = as.integer(n_neighbors),
      min_dist = min_dist,
      metric = metric
    )
  )
}

#' t-SNE embedding
#'
#' t-SNE currently uses the CPU `Rtsne` backend.
#'
#' @inheritParams cuda_umap
#' @param perplexity t-SNE perplexity.
#' @param theta Barnes-Hut accuracy/speed trade-off.
#' @param ... Additional arguments passed to `Rtsne::Rtsne()`.
#' @return A `cuda_embedding`; see [cuda_umap()] for the stable result fields.
#' @export
#' @examples
#' if (requireNamespace("Rtsne", quietly = TRUE)) {
#'   cuda_tsne(matrix(rnorm(120), 40, 3), perplexity = 5, seed = 1)
#' }
cuda_tsne <- function(x, n_components = 2L, perplexity = 30,
                      theta = 0.5, seed = NULL, ...) {
  input <- .embedding_input(x)
  n_components <- .embedding_components(
    n_components,
    min(3L, nrow(input$matrix) - 2L)
  )
  if (!is.numeric(perplexity) || length(perplexity) != 1L ||
      is.na(perplexity) || !is.finite(perplexity) || perplexity <= 0 ||
      3 * perplexity >= nrow(input$matrix) - 1L) {
    stop("`perplexity` must satisfy 3 * perplexity < nrow(x) - 1.",
         call. = FALSE)
  }
  if (!is.numeric(theta) || length(theta) != 1L || is.na(theta) ||
      !is.finite(theta) || theta < 0 || theta > 1) {
    stop("`theta` must be between 0 and 1.", call. = FALSE)
  }
  if (!requireNamespace("Rtsne", quietly = TRUE)) {
    stop("Install the 'Rtsne' package to compute t-SNE embeddings.",
         call. = FALSE)
  }
  fit <- .with_embedding_seed(
    seed,
    Rtsne::Rtsne(
      input$matrix,
      dims = n_components,
      perplexity = perplexity,
      theta = theta,
      pca = FALSE,
      check_duplicates = FALSE,
      verbose = FALSE,
      ...
    )
  )
  .new_embedding(
    fit$Y,
    method = "tsne",
    backend = "Rtsne",
    input = input,
    parameters = list(perplexity = perplexity, theta = theta)
  )
}

#' Diffusion-map-style embedding
#'
#' Pairwise distances can use the `cudalearnr` CUDA path. Kernel construction
#' and eigendecomposition currently run on the CPU.
#'
#' @param x Numeric observation-by-feature matrix or compatible cudaverse result.
#' @param n_components Output dimensions.
#' @param sigma Gaussian kernel bandwidth. Defaults to the median positive
#'   pairwise distance.
#' @param diffusion_time Non-negative diffusion time exponent.
#' @param metric Euclidean or cosine distance.
#' @param device Device passed to [cudalearnr::cuda_distance()].
#' @return A `cuda_embedding` with the stable fields documented by
#'   [cuda_umap()], stage-level distance/kernel/eigendecomposition provenance,
#'   and an additional `eigenvalues` element.
#' @export
#' @examples
#' cuda_diffusion_map(
#'   matrix(rnorm(120), 40, 3),
#'   n_components = 2,
#'   device = "cpu"
#' )
cuda_diffusion_map <- function(x, n_components = 2L, sigma = NULL,
                               diffusion_time = 1,
                               metric = c("euclidean", "cosine"),
                               device = c("auto", "cuda", "cpu")) {
  input <- .embedding_input(x)
  n_components <- .embedding_components(
    n_components,
    nrow(input$matrix) - 2L
  )
  if (!is.numeric(diffusion_time) || length(diffusion_time) != 1L ||
      is.na(diffusion_time) || !is.finite(diffusion_time) ||
      diffusion_time < 0) {
    stop("`diffusion_time` must be finite and non-negative.",
         call. = FALSE)
  }
  metric <- match.arg(metric)
  distances <- cudalearnr::cuda_distance(
    input$matrix,
    metric = metric,
    device = device
  )
  positive <- distances[distances > 0 & is.finite(distances)]
  if (is.null(sigma)) {
    sigma <- if (length(positive)) stats::median(positive) else 1
  }
  if (!is.numeric(sigma) || length(sigma) != 1L || is.na(sigma) ||
      !is.finite(sigma) || sigma <= 0) {
    stop("`sigma` must be a positive finite number.", call. = FALSE)
  }
  kernel <- exp(-(distances^2) / (2 * sigma^2))
  diag(kernel) <- 0
  degree <- rowSums(kernel)
  if (any(degree <= 0)) {
    stop("The diffusion kernel contains isolated observations.",
         call. = FALSE)
  }
  inverse_root_degree <- 1 / sqrt(degree)
  normalized <- kernel * tcrossprod(inverse_root_degree)
  eigen_count <- n_components + 1L
  if (nrow(normalized) > 500L &&
      requireNamespace("RSpectra", quietly = TRUE)) {
    decomposition <- RSpectra::eigs_sym(
      normalized,
      k = eigen_count,
      which = "LA"
    )
    order_index <- order(decomposition$values, decreasing = TRUE)
    values <- decomposition$values[order_index]
    vectors <- decomposition$vectors[, order_index, drop = FALSE]
    backend <- "RSpectra"
  } else {
    decomposition <- eigen(normalized, symmetric = TRUE)
    values <- decomposition$values[seq_len(eigen_count)]
    vectors <- decomposition$vectors[, seq_len(eigen_count), drop = FALSE]
    backend <- "base-eigen"
  }
  retained_values <- pmax(values[-1L], 0)
  coordinates <- (
    vectors[, -1L, drop = FALSE] * inverse_root_degree
  ) * rep(retained_values^diffusion_time, each = nrow(vectors))
  distance_device <- attr(distances, "device") %||% "cpu"
  compute_device <- if (identical(distance_device, "cpu")) {
    "cpu"
  } else {
    "hybrid"
  }
  result <- .new_embedding(
    coordinates,
    method = "diffusion",
    backend = backend,
    input = input,
    parameters = list(
      sigma = sigma,
      diffusion_time = diffusion_time,
      metric = metric
    ),
    compute_device = compute_device,
    compute_stages = list(
      distance = list(
        device = distance_device,
        backend = if (identical(distance_device, "cuda")) "torch" else "base"
      ),
      kernel = list(device = "cpu", backend = "base"),
      eigendecomposition = list(device = "cpu", backend = backend)
    )
  )
  result$eigenvalues <- retained_values
  result
}

#' Extract embedding coordinates
#'
#' @param x A `cuda_embedding`.
#' @return Numeric coordinate matrix.
#' @export
#' @examples
#' fit <- cuda_diffusion_map(
#'   matrix(rnorm(60), 20, 3),
#'   n_components = 2,
#'   device = "cpu"
#' )
#' embedding_coordinates(fit)
embedding_coordinates <- function(x) {
  if (!inherits(x, "cuda_embedding")) {
    stop("`x` must be a cuda_embedding object.", call. = FALSE)
  }
  x$coordinates
}

#' @export
print.cuda_embedding <- function(x, ...) {
  cat(sprintf(
    "<cuda_embedding method=%s observations=%s dimensions=%s backend=%s compute_device=%s>\n",
    x$method, nrow(x$coordinates), ncol(x$coordinates), x$backend,
    x$compute_device
  ))
  invisible(x)
}
