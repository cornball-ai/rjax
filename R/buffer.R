#' Transfer data to an XLA device buffer
#'
#' Copies an R numeric vector to device memory.
#'
#' @param client An \code{xla_client} object.
#' @param data A numeric or integer vector.
#' @param dims Integer vector of dimensions. Defaults to \code{length(data)}.
#' @param dtype Character: \code{"f32"}, \code{"f64"}, \code{"i32"}.
#'   Defaults to \code{"f32"}.
#' @return An external pointer of class \code{xla_buffer}.
#' @export
xla_buffer <- function(client, data, dims = NULL, dtype = "f32") {
  if (!inherits(client, "xla_client")) stop("expected an xla_client object")
  if (is.null(dims)) dims <- infer_dims(data)
  dims <- as.integer(dims)
  dtype <- match.arg(dtype, c("f32", "f64", "i32"))
  ptr <- rjax_buffer_from_r(client, as.numeric(data), dims, dtype)
  class(ptr) <- "xla_buffer"
  ptr
}

#' @export
print.xla_buffer <- function(x, ...) {
  info <- rjax_buffer_shape(x)
  cat("<xla_buffer:", info$dtype,
      paste0("[", paste(info$dims, collapse = ", "), "]"), ">\n")
  invisible(x)
}

#' @export
as.array.xla_buffer <- function(x, ...) {
  rjax_buffer_to_r(x)
}

#' @export
as.numeric.xla_buffer <- function(x, ...) {
  as.numeric(rjax_buffer_to_r(x))
}
