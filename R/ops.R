# S3 operator overloading for xla_op
#
# Allows writing natural R math that builds XLA computation graphs:
#   2 * x + 1
#   exp(-2 * x)
#   (1 - y) / (1 + y)

# Helper: apply a binary op, handling scalar promotion and tape recording
binary_op <- function(op_name, cpp_fn, e1, e2) {
  if (inherits(e1, "xla_op")) {
    e2 <- ensure_xla_op(e2, e1)
  } else {
    e1 <- ensure_xla_op(e1, e2)
  }
  b <- attr(e1, "builder")
  if (is.null(b)) b <- attr(e2, "builder")
  result <- tag_op(cpp_fn(e1, e2), b)
  tape_record(op_name, result, list(e1, e2))
}

# Helper: apply a unary op with tape recording
unary_op <- function(op_name, cpp_fn, x) {
  b <- attr(x, "builder")
  result <- tag_op(cpp_fn(x), b)
  tape_record(op_name, result, list(x))
}

#' @export
`+.xla_op` <- function(e1, e2) {
  if (missing(e2)) return(e1)  # unary +
  binary_op("add", rjax_add, e1, e2)
}

#' @export
`-.xla_op` <- function(e1, e2) {
  if (missing(e2)) return(unary_op("neg", rjax_neg, e1))
  binary_op("sub", rjax_sub, e1, e2)
}

#' @export
`*.xla_op` <- function(e1, e2) binary_op("mul", rjax_mul, e1, e2)

#' @export
`/.xla_op` <- function(e1, e2) binary_op("div", rjax_div, e1, e2)

#' @export
`^.xla_op` <- function(e1, e2) binary_op("pow", rjax_pow, e1, e2)

# Math group generic for xla_op
#' @export
Math.xla_op <- function(x, ...) {
  fn <- switch(.Generic,
    exp   = function(x) unary_op("exp", rjax_exp, x),
    log   = function(x) unary_op("log", rjax_log, x),
    sqrt  = function(x) unary_op("sqrt", rjax_sqrt, x),
    abs   = function(x) unary_op("abs", rjax_abs, x),
    sin   = function(x) unary_op("sin", rjax_sin, x),
    cos   = function(x) unary_op("cos", rjax_cos, x),
    tanh  = function(x) unary_op("tanh", rjax_tanh, x),
    stop("Math operation '", .Generic, "' not supported for xla_op")
  )
  fn(x)
}

# sum() for xla_op: reduce sum over all dimensions
#' @export
Summary.xla_op <- function(..., na.rm = FALSE) {
  if (.Generic != "sum") stop(.Generic, " not supported for xla_op")
  x <- ..1
  b <- attr(x, "builder")
  sh <- op_shape(x)
  dims <- seq(0L, length(sh$dims) - 1L)
  result <- tag_op(rjax_reduce_sum(x, as.integer(dims), b), b)
  tape_record("reduce_sum", result, list(x), extras = list(reduce_dims = dims))
}
