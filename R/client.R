#' Create an XLA client
#'
#' Creates a PJRT client for the loaded plugin backend.
#' The PJRT plugin must be loaded first (either via PJRT_PLUGIN_PATH
#' environment variable or by calling the plugin loader directly).
#'
#' @return An external pointer of class \code{xla_client}.
#' @export
xla_client <- function() {
  .Call(rjax_client_create)
}

#' List devices on a client
#'
#' @param client An \code{xla_client} object.
#' @return Character vector of device descriptions.
#' @export
xla_devices <- function(client) {
  if (!inherits(client, "xla_client")) {
    stop("expected an xla_client object")
  }
  .Call(rjax_client_devices, client)
}

#' @export
print.xla_client <- function(x, ...) {
  cat("<xla_client>\n")
  invisible(x)
}
