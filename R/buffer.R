#' Transfer data to an XLA device buffer
#'
#' Copies an R numeric vector to device memory managed by the PJRT client.
#'
#' @param client An \code{xla_client} object.
#' @param data A numeric or integer vector.
#' @param dims Integer vector of dimensions. Defaults to \code{length(data)}
#'   (a 1-D tensor).
#' @param dtype Character scalar: \code{"f32"}, \code{"f64"}, \code{"i32"}.
#'   Defaults to \code{"f32"}.
#' @return An external pointer of class \code{xla_buffer}.
#' @export
xla_buffer <- function(client, data, dims = length(data), dtype = "f32") {
  if (!inherits(client, "xla_client")) {
    stop("expected an xla_client object")
  }
  dims <- as.integer(dims)
  dtype <- match.arg(dtype, c("f32", "f64", "i32"))
  .Call(rjax_buffer_from_r, client, data, dims, dtype)
}

#' @export
print.xla_buffer <- function(x, ...) {
  cat("<xla_buffer>\n")
  invisible(x)
}

#' @export
as.array.xla_buffer <- function(x, ...) {
  .Call(rjax_buffer_to_r, x)
}
