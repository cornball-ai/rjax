# Test reverse-mode automatic differentiation

# d/dx exp(x) = exp(x)
gexp <- xla_grad(exp)
expect_equal(gexp(0.0), 1.0, tolerance = 1e-5)
expect_equal(gexp(1.0), exp(1), tolerance = 1e-4)

# d/dx sin(x) = cos(x)
gsin <- xla_grad(sin)
expect_equal(gsin(0.0), 1.0, tolerance = 1e-5)
expect_equal(gsin(pi/2), cos(pi/2), tolerance = 1e-4)

# d/dx x^3 = 3x^2
g_cube <- xla_grad(function(x) x^3)
expect_equal(g_cube(2.0), 12.0, tolerance = 1e-4)
expect_equal(g_cube(3.0), 27.0, tolerance = 1e-4)

# d/dx log(x) = 1/x
glog <- xla_grad(log)
expect_equal(glog(1.0), 1.0, tolerance = 1e-5)
expect_equal(glog(2.0), 0.5, tolerance = 1e-5)

# d/dx tanh(x) = 1 - tanh(x)^2 (built-in tanh)
gtanh <- xla_grad(tanh)
expect_equal(gtanh(0.0), 1.0, tolerance = 1e-5)
expect_equal(gtanh(1.0), 1 - tanh(1)^2, tolerance = 1e-4)

# JAX tanh example: custom implementation
tanh_custom <- function(x) {
  y <- exp(-2 * x)
  (1 - y) / (1 + y)
}
grad_tanh <- xla_grad(tanh_custom)
expect_equal(grad_tanh(1.0), 1 - tanh(1)^2, tolerance = 1e-4)

# Vector -> scalar: d/dw sum(w^2) = 2w
gloss <- xla_grad(function(w) sum(w^2))
expect_equal(gloss(c(1, 2, 3)), c(2, 4, 6), tolerance = 1e-4)

# Multi-argument: grad w.r.t. first arg
# d/dx (x*y + sin(x)) = y + cos(x)
gf <- xla_grad(function(x, y) x * y + sin(x))
expect_equal(gf(0.0, 3.0), 4.0, tolerance = 1e-5)  # 3 + cos(0) = 4

# softplus: d/dx log(1+exp(x)) = sigmoid(x)
gsoftplus <- xla_grad(function(x) log(1 + exp(x)))
expect_equal(gsoftplus(0.0), 0.5, tolerance = 1e-5)

# ---- Grad of grad (higher-order derivatives) ----

# d2/dx2 x^3 = 6x
g2_cube <- xla_grad(xla_grad(function(x) x^3))
expect_equal(g2_cube(2.0), 12.0, tolerance = 1e-4)

# d2/dx2 sin(x) = -sin(x)
g2_sin <- xla_grad(xla_grad(sin))
expect_equal(g2_sin(0.0), 0.0, tolerance = 1e-5)
expect_equal(g2_sin(pi / 2), -1.0, tolerance = 1e-4)

# d3/dx3 x^4 = 24x
g3_x4 <- xla_grad(xla_grad(xla_grad(function(x) x^4)))
expect_equal(g3_x4(1.0), 24.0, tolerance = 1e-3)

