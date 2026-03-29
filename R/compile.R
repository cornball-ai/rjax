#' Compile an HLO module
#'
#' Takes serialized HLO module bytes (protobuf) and compiles them
#' into an executable for the client's backend.
#'
#' @param client An \code{xla_client} object.
#' @param hlo_bytes A raw vector containing a serialized HLO module proto.
#' @return An external pointer of class \code{xla_executable}.
#' @export
xla_compile <- function(client, hlo_bytes) {
  if (!inherits(client, "xla_client")) {
    stop("expected an xla_client object")
  }
  if (!is.raw(hlo_bytes)) {
    stop("hlo_bytes must be a raw vector")
  }
  .Call(rjax_compile, client, hlo_bytes)
}

#' @export
print.xla_executable <- function(x, ...) {
  cat("<xla_executable>\n")
  invisible(x)
}
