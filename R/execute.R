#' Execute a compiled XLA program
#'
#' Runs a compiled executable with the given input buffers.
#'
#' @param executable An \code{xla_executable} object.
#' @param inputs A list of \code{xla_buffer} objects.
#' @return A list of \code{xla_buffer} objects (one per output).
#' @export
xla_execute <- function(executable, inputs = list()) {
  if (!inherits(executable, "xla_executable")) {
    stop("expected an xla_executable object")
  }
  results <- rjax_execute(executable, inputs)
  lapply(results, function(ptr) { class(ptr) <- "xla_buffer"; ptr })
}
