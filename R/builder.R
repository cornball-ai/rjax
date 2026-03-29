#' Create an XLA computation builder
#'
#' @param name A name for the computation (used in debug output).
#' @return An external pointer of class \code{xla_builder}.
#' @export
xla_builder <- function(name = "computation") {
  ptr <- rjax_builder_new(name)
  class(ptr) <- "xla_builder"
  ptr
}

#' Add a parameter to the computation
#'
#' @param builder An \code{xla_builder} object.
#' @param param_num Zero-based parameter index.
#' @param dtype Character: \code{"f32"}, \code{"f64"}, \code{"i32"}.
#' @param dims Integer vector of dimensions.
#' @return An \code{xla_op} external pointer.
#' @export
xla_parameter <- function(builder, param_num, dtype = "f32", dims = integer(0)) {
  if (!inherits(builder, "xla_builder")) stop("expected an xla_builder object")
  ptr <- rjax_parameter(builder, as.integer(param_num), dtype, as.integer(dims))
  class(ptr) <- "xla_op"
  ptr
}

#' Create a constant vector
#'
#' @param builder An \code{xla_builder} object.
#' @param values Numeric vector of values.
#' @param dtype Character: \code{"f32"}, \code{"f64"}, \code{"i32"}.
#' @return An \code{xla_op} external pointer.
#' @export
xla_constant <- function(builder, values, dtype = "f32") {
  if (!inherits(builder, "xla_builder")) stop("expected an xla_builder object")
  if (length(values) == 1) {
    ptr <- rjax_constant_scalar(builder, as.numeric(values), dtype)
  } else {
    ptr <- rjax_constant_r1(builder, as.numeric(values), dtype)
  }
  class(ptr) <- "xla_op"
  ptr
}

# ---- Arithmetic ops ----

#' Element-wise addition
#' @param lhs An \code{xla_op}.
#' @param rhs An \code{xla_op}.
#' @return An \code{xla_op}.
#' @export
xla_add <- function(lhs, rhs) {
  ptr <- rjax_add(lhs, rhs)
  class(ptr) <- "xla_op"
  ptr
}

#' Element-wise subtraction
#' @param lhs An \code{xla_op}.
#' @param rhs An \code{xla_op}.
#' @return An \code{xla_op}.
#' @export
xla_sub <- function(lhs, rhs) {
  ptr <- rjax_sub(lhs, rhs)
  class(ptr) <- "xla_op"
  ptr
}

#' Element-wise multiplication
#' @param lhs An \code{xla_op}.
#' @param rhs An \code{xla_op}.
#' @return An \code{xla_op}.
#' @export
xla_mul <- function(lhs, rhs) {
  ptr <- rjax_mul(lhs, rhs)
  class(ptr) <- "xla_op"
  ptr
}

#' Element-wise division
#' @param lhs An \code{xla_op}.
#' @param rhs An \code{xla_op}.
#' @return An \code{xla_op}.
#' @export
xla_div <- function(lhs, rhs) {
  ptr <- rjax_div(lhs, rhs)
  class(ptr) <- "xla_op"
  ptr
}

#' Negation
#' @param x An \code{xla_op}.
#' @return An \code{xla_op}.
#' @export
xla_neg <- function(x) {
  ptr <- rjax_neg(x)
  class(ptr) <- "xla_op"
  ptr
}

#' Absolute value
#' @param x An \code{xla_op}.
#' @return An \code{xla_op}.
#' @export
xla_abs <- function(x) {
  ptr <- rjax_abs(x)
  class(ptr) <- "xla_op"
  ptr
}

#' Exponential
#' @param x An \code{xla_op}.
#' @return An \code{xla_op}.
#' @export
xla_exp <- function(x) {
  ptr <- rjax_exp(x)
  class(ptr) <- "xla_op"
  ptr
}

#' Natural log
#' @param x An \code{xla_op}.
#' @return An \code{xla_op}.
#' @export
xla_log <- function(x) {
  ptr <- rjax_log(x)
  class(ptr) <- "xla_op"
  ptr
}

#' Square root
#' @param x An \code{xla_op}.
#' @return An \code{xla_op}.
#' @export
xla_sqrt <- function(x) {
  ptr <- rjax_sqrt(x)
  class(ptr) <- "xla_op"
  ptr
}

#' Hyperbolic tangent
#' @param x An \code{xla_op}.
#' @return An \code{xla_op}.
#' @export
xla_tanh <- function(x) {
  ptr <- rjax_tanh(x)
  class(ptr) <- "xla_op"
  ptr
}

# ---- Matrix ops ----

#' Dot product / matrix multiply
#' @param lhs An \code{xla_op}.
#' @param rhs An \code{xla_op}.
#' @return An \code{xla_op}.
#' @export
xla_dot <- function(lhs, rhs) {
  ptr <- rjax_dot(lhs, rhs)
  class(ptr) <- "xla_op"
  ptr
}

#' Transpose
#' @param x An \code{xla_op}.
#' @param permutation Integer vector of dimension permutation (0-indexed).
#' @return An \code{xla_op}.
#' @export
xla_transpose <- function(x, permutation) {
  ptr <- rjax_transpose(x, as.integer(permutation))
  class(ptr) <- "xla_op"
  ptr
}

#' Reshape
#' @param x An \code{xla_op}.
#' @param new_dims Integer vector of new dimensions.
#' @return An \code{xla_op}.
#' @export
xla_reshape <- function(x, new_dims) {
  ptr <- rjax_reshape(x, as.integer(new_dims))
  class(ptr) <- "xla_op"
  ptr
}

# ---- Build ----

#' Build a computation from the builder
#'
#' Builds the XLA computation graph. The last operation added becomes the
#' root (output) of the computation, unless \code{root} is specified.
#'
#' @param builder An \code{xla_builder} object.
#' @param root Optional \code{xla_op} to use as the root. If NULL, uses
#'   the last operation.
#' @return An \code{xla_computation} external pointer.
#' @export
xla_build <- function(builder, root = NULL) {
  if (!inherits(builder, "xla_builder")) stop("expected an xla_builder object")
  if (is.null(root)) {
    ptr <- rjax_build(builder)
  } else {
    ptr <- rjax_build_with_root(builder, root)
  }
  class(ptr) <- "xla_computation"
  ptr
}

#' @export
print.xla_op <- function(x, ...) {
  cat("<xla_op>\n")
  invisible(x)
}

#' @export
print.xla_computation <- function(x, ...) {
  cat("<xla_computation>\n")
  invisible(x)
}
