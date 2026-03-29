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
xla_jit <- function(f, dtype = "f32") {
  cache <- new.env(parent = emptyenv())
  client <- NULL

  function(...) {
    if (is.null(client)) client <<- xla_client()
    args <- list(...)

    # Cache key based on shapes
    key <- paste(vapply(args, function(a) {
      paste(length(a), dtype, sep = ":")
    }, character(1)), collapse = ",")

    if (is.null(cache[[key]])) {
      # Trace: create builder, params, run f, build, compile
      builder <- xla_builder("jit")
      params <- lapply(seq_along(args), function(i) {
        d <- as.integer(if (length(args[[i]]) == 1) integer(0) else length(args[[i]]))
        p <- rjax_parameter(builder, i - 1L, dtype, d)
        tag_op(p, builder)
      })
      result <- do.call(f, params)
      comp <- xla_build(builder, result)
      cache[[key]] <- list(
        executable = xla_compile(client, comp),
        n_args = length(args)
      )
    }

    # Execute
    bufs <- lapply(args, function(a) {
      d <- as.integer(if (length(a) == 1) integer(0) else length(a))
      xla_buffer(client, as.numeric(a), d, dtype)
    })
    out <- xla_execute(cache[[key]]$executable, bufs)
    as.array(out[[1]])
  }
}
