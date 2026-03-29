#' Create an XLA client
#'
#' Creates a CPU-backed XLA client for compiling and executing computations.
#'
#' @return An external pointer of class \code{xla_client}.
#' @export
xla_client <- function(backend = "cpu") {
  backend <- match.arg(backend, c("cpu", "gpu", "cuda"))
  ptr <- rjax_client_create(backend)
  class(ptr) <- "xla_client"
  ptr
}

#' List devices on a client
#'
#' @param client An \code{xla_client} object.
#' @return Character vector of device descriptions.
#' @export
xla_devices <- function(client) {
  if (!inherits(client, "xla_client")) stop("expected an xla_client object")
  rjax_client_devices(client)
}

#' @export
print.xla_client <- function(x, ...) {
  platform <- rjax_client_platform(x)
  devices <- rjax_client_devices(x)
  cat("<xla_client:", platform, "with", length(devices), "device(s)>\n")
  invisible(x)
}
