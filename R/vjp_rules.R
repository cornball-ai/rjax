# VJP (vector-Jacobian product) rules for reverse-mode autodiff.
#
# Each rule takes: g (upstream gradient), inputs (list of forward inputs),
# output (forward output). Returns: list of gradients for each input.
# All values are xla_ops on the same builder.

.vjp_rules <- list(
  # Parameters are leaf nodes, no further backprop needed
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
    b <- attr(g, "builder")
    s <- tag_op(rjax_sign(inputs[[1]]), b)
    list(g * s)
  },

  exp = function(g, inputs, output) {
    # reuse forward output: d/dx exp(x) = exp(x)
    list(g * output)
  },

  log = function(g, inputs, output) {
    list(g / inputs[[1]])
  },

  sqrt = function(g, inputs, output) {
    # d/dx sqrt(x) = 1/(2*sqrt(x)), reuse output
    list(g / (output + output))
  },

  tanh = function(g, inputs, output) {
    # d/dx tanh(x) = 1 - tanh(x)^2, reuse output
    list(g * (1 - output * output))
  },

  sin = function(g, inputs, output) {
    b <- attr(g, "builder")
    list(g * tag_op(rjax_cos(inputs[[1]]), b))
  },

  cos = function(g, inputs, output) {
    b <- attr(g, "builder")
    list(-g * tag_op(rjax_sin(inputs[[1]]), b))
  },

  pow = function(g, inputs, output) {
    a <- inputs[[1]]
    b_val <- inputs[[2]]
    # d/da a^b = b * a^(b-1)
    # d/db a^b = log(a) * a^b
    b_env <- attr(g, "builder")
    list(
      g * b_val * (a ^ (b_val - 1)),
      g * tag_op(rjax_log(a), b_env) * output
    )
  },

  dot = function(g, inputs, output) {
    a <- inputs[[1]]
    b_val <- inputs[[2]]
    bld <- attr(g, "builder")
    # For 1-D dot products: grad_a = g * b, grad_b = g * a
    # For matrix multiply: grad_a = g @ b^T, grad_b = a^T @ g
    a_sh <- op_shape(a)
    b_sh <- op_shape(b_val)
    if (length(a_sh$dims) <= 1 && length(b_sh$dims) <= 1) {
      list(g * b_val, g * a)
    } else {
      b_t <- tag_op(rjax_transpose(b_val, as.integer(rev(seq_along(b_sh$dims) - 1L))), bld)
      a_t <- tag_op(rjax_transpose(a, as.integer(rev(seq_along(a_sh$dims) - 1L))), bld)
      list(
        tag_op(rjax_dot(g, b_t), bld),
        tag_op(rjax_dot(a_t, g), bld)
      )
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
    # Gradient of sum is broadcast of g back to input shape
    in_shape <- op_shape(inputs[[1]])
    bld <- attr(g, "builder")
    # g is a scalar (or reduced tensor), broadcast back
    expanded <- tag_op(
      rjax_constant_broadcast(bld, 1.0, in_shape$dtype, as.integer(in_shape$dims)),
      bld
    )
    list(g * expanded)
  }
)
