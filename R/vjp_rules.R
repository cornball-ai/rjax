# VJP (vector-Jacobian product) rules for reverse-mode autodiff.
#
# Each rule takes: g (upstream gradient), inputs (list of forward inputs),
# output (forward output). Returns: list of gradients for each input.
#
# All operations use the overloaded operators (+, -, *, /) or the
# traced_unary/traced_binary helpers so they record on the active tape.

# Helpers for VJP rules: create ops that record on the current tape.
traced_unary <- function(op_name, cpp_fn, x) {
  b <- attr(x, "builder")
  if (is.null(b)) b <- op_builder(x)
  result <- tag_op(cpp_fn(x), b)
  tape_record(op_name, result, list(x))
}

.vjp_rules <- list(
  param = function(g, inputs, output) {
    list()
  },

  add = function(g, inputs, output) {
    list(g, g)
  },

  sub = function(g, inputs, output) {
    list(g, -g)
  },

  mul = function(g, inputs, output) {
    list(g * inputs[[2]], g * inputs[[1]])
  },

  div = function(g, inputs, output) {
    a <- inputs[[1]]
    b <- inputs[[2]]
    list(g / b, -g * a / (b * b))
  },

  neg = function(g, inputs, output) {
    list(-g)
  },

  abs = function(g, inputs, output) {
    list(g * traced_unary("sign", rjax_sign, inputs[[1]]))
  },

  exp = function(g, inputs, output) {
    list(g * output)
  },

  log = function(g, inputs, output) {
    list(g / inputs[[1]])
  },

  sqrt = function(g, inputs, output) {
    list(g / (output + output))
  },

  tanh = function(g, inputs, output) {
    list(g * (1 - output * output))
  },

  sin = function(g, inputs, output) {
    list(g * traced_unary("cos", rjax_cos, inputs[[1]]))
  },

  cos = function(g, inputs, output) {
    list(-g * traced_unary("sin", rjax_sin, inputs[[1]]))
  },

  pow = function(g, inputs, output) {
    a <- inputs[[1]]
    b <- inputs[[2]]
    list(
      g * b * (a ^ (b - 1)),
      g * traced_unary("log", rjax_log, a) * output
    )
  },

  dot = function(g, inputs, output) {
    a <- inputs[[1]]
    b <- inputs[[2]]
    bld <- attr(g, "builder")
    a_sh <- op_shape(a)
    b_sh <- op_shape(b)
    if (length(a_sh$dims) <= 1 && length(b_sh$dims) <= 1) {
      list(g * b, g * a)
    } else {
      b_perm <- as.integer(rev(seq_along(b_sh$dims) - 1L))
      a_perm <- as.integer(rev(seq_along(a_sh$dims) - 1L))
      b_t <- tag_op(rjax_transpose(b, b_perm), bld)
      b_t <- tape_record("transpose", b_t, list(b), extras = list(permutation = b_perm))
      a_t <- tag_op(rjax_transpose(a, a_perm), bld)
      a_t <- tape_record("transpose", a_t, list(a), extras = list(permutation = a_perm))
      ga <- tag_op(rjax_dot(g, b_t), bld)
      ga <- tape_record("dot", ga, list(g, b_t))
      gb <- tag_op(rjax_dot(a_t, g), bld)
      gb <- tape_record("dot", gb, list(a_t, g))
      list(ga, gb)
    }
  },

  transpose = function(g, inputs, output) {
    perm <- output$extras$permutation
    inv_perm <- integer(length(perm))
    for (i in seq_along(perm)) inv_perm[perm[i] + 1L] <- i - 1L
    bld <- attr(g, "builder")
    list(tag_op(rjax_transpose(g, as.integer(inv_perm)), bld))
  },

  reshape = function(g, inputs, output) {
    orig_shape <- op_shape(inputs[[1]])
    bld <- attr(g, "builder")
    list(tag_op(rjax_reshape(g, as.integer(orig_shape$dims)), bld))
  },

  reduce_sum = function(g, inputs, output) {
    in_shape <- op_shape(inputs[[1]])
    bld <- attr(g, "builder")
    expanded <- tag_op(
      rjax_constant_broadcast(bld, 1.0, in_shape$dtype, as.integer(in_shape$dims)),
      bld
    )
    list(g * expanded)
  }
)
