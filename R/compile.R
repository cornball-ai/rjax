#' Compile an XLA computation
#'
#' Compiles an XlaComputation into an executable for the client's backend.
#'
#' @param client An \code{xla_client} object.
#' @param computation An \code{xla_computation} object from \code{xla_build}.
#' @return An external pointer of class \code{xla_executable}.
#' @export
xla_compile <- function(client, computation) {
  if (!inherits(client, "xla_client")) stop("expected an xla_client object")
  if (!inherits(computation, "xla_computation")) {
    stop("expected an xla_computation object")
  }
  ptr <- rjax_compile(client, computation)
  class(ptr) <- "xla_executable"
  ptr
}

#' @export
print.xla_executable <- function(x, ...) {
  cat("<xla_executable>\n")
  invisible(x)
}
