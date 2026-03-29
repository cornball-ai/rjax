# Test JIT compilation

# Basic scalar function
f <- function(x) exp(x) + 1
jf <- xla_jit(f)
expect_equal(jf(0.0), 2.0, tolerance = 1e-5)
expect_equal(jf(1.0), exp(1) + 1, tolerance = 1e-5)

# Vector function
jf2 <- xla_jit(function(x) 2 * x + 1)
expect_equal(jf2(c(1, 2, 3)), c(3, 5, 7), tolerance = 1e-5)

# Cache reuse (same shapes)
expect_equal(jf2(c(4, 5, 6)), c(9, 11, 13), tolerance = 1e-5)

# Multi-argument
add_fn <- function(x, y) x + y
jadd <- xla_jit(add_fn)
expect_equal(jadd(c(1, 2), c(3, 4)), c(4, 6), tolerance = 1e-5)

# Compound expression
tanh_custom <- function(x) {
  y <- exp(-2 * x)
  (1 - y) / (1 + y)
}
jtanh <- xla_jit(tanh_custom)
expect_equal(jtanh(0.0), 0.0, tolerance = 1e-5)
expect_equal(jtanh(1.0), tanh(1), tolerance = 1e-4)
