# Shape inference and broadcast helpers

# Get the builder pointer from an xla_op
op_builder <- function(op) {
  rjax_op_builder(op)
}

# Get shape (dims + dtype) of an xla_op
op_shape <- function(op) {
  b <- attr(op, "builder")
  if (is.null(b)) b <- op_builder(op)
  rjax_op_shape(b, op)
}

# Wrap a numeric scalar/vector as an xla_constant, matching the builder of `ref_op`
ensure_xla_op <- function(x, ref_op) {
  if (inherits(x, "xla_op")) return(x)
  b <- attr(ref_op, "builder")
  if (is.null(b)) b <- op_builder(ref_op)
  dtype <- op_shape(ref_op)$dtype
  if (length(x) == 1) {
    out <- rjax_constant_scalar(b, as.numeric(x), dtype)
  } else {
    out <- rjax_constant_r1(b, as.numeric(x), dtype)
  }
  class(out) <- "xla_op"
  attr(out, "builder") <- b
  out
}

# Tag an xla_op with its builder (for operator overloading chain)
tag_op <- function(op, builder) {
  class(op) <- "xla_op"
  attr(op, "builder") <- builder
  op
}
