#' JIT-compile an R function for XLA execution
#'
#' Returns a new function that traces the input function on first call,
#' compiles it with XLA, and caches the result. Subsequent calls with
#' the same input shapes reuse the cached compiled executable.
#'
#' @param f An R function using arithmetic ops (+, -, *, /, ^),
#'   math functions (exp, log, sqrt, sin, cos, tanh), and sum().
#' @param dtype Default dtype for inputs: "f32" or "f64".
#' @return A function that accepts numeric inputs and returns numeric output.
#' @export
xla_jit <- function(f, dtype = "f32", backend = "cpu") {
  cache <- new.env(parent = emptyenv())
  client <- NULL

  function(...) {
    if (is.null(client)) client <<- xla_client(backend)
    args <- list(...)

    key <- paste(vapply(args, function(a) {
      paste(paste(infer_dims(a), collapse = "x"), dtype, sep = ":")
    }, character(1)), collapse = ",")

    if (is.null(cache[[key]])) {
      builder <- xla_builder("jit")
      params <- lapply(seq_along(args), function(i) {
        d <- infer_dims(args[[i]])
        p <- rjax_parameter(builder, i - 1L, dtype, d)
        tag_op(p, builder)
      })
      result <- do.call(f, params)
      comp <- xla_build(builder, result)
      cache[[key]] <- xla_compile(client, comp)
    }

    bufs <- lapply(args, function(a) {
      d <- infer_dims(a)
      xla_buffer(client, as.numeric(a), d, dtype)
    })
    out <- xla_execute(cache[[key]], bufs)
    as.array(out[[1]])
  }
}
