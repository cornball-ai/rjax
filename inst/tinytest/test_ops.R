# Test operator overloading on xla_op

client <- xla_client()

# Helper: build, compile, execute a single-param computation
run1 <- function(build_fn, input, dtype = "f32") {
  builder <- xla_builder("test")
  dims <- as.integer(if (length(input) == 1) integer(0) else length(input))
  x <- xla_parameter(builder, 0L, dtype, dims)
  attr(x, "builder") <- builder
  class(x) <- "xla_op"
  result <- build_fn(x)
  comp <- xla_build(builder, result)
  exec <- xla_compile(client, comp)
  buf <- xla_buffer(client, as.numeric(input), dims, dtype)
  as.array(xla_execute(exec, list(buf))[[1]])
}

# Arithmetic
expect_equal(run1(function(x) x + 1, c(1, 2, 3)), c(2, 3, 4), tolerance = 1e-5)
expect_equal(run1(function(x) 2 * x, c(1, 2, 3)), c(2, 4, 6), tolerance = 1e-5)
expect_equal(run1(function(x) x / 2, c(4, 6, 8)), c(2, 3, 4), tolerance = 1e-5)
expect_equal(run1(function(x) x - 1, c(5, 6, 7)), c(4, 5, 6), tolerance = 1e-5)
expect_equal(run1(function(x) -x, c(1, -2, 3)), c(-1, 2, -3), tolerance = 1e-5)
expect_equal(run1(function(x) x^2, c(2, 3)), c(4, 9), tolerance = 1e-5)

# Math group generic
expect_equal(run1(exp, c(0, 1)), c(1, exp(1)), tolerance = 1e-5)
expect_equal(run1(log, c(1, exp(1))), c(0, 1), tolerance = 1e-5)
expect_equal(run1(sqrt, c(4, 9)), c(2, 3), tolerance = 1e-5)
expect_equal(run1(abs, c(-1, 2, -3)), c(1, 2, 3), tolerance = 1e-5)
expect_equal(run1(sin, c(0)), c(0), tolerance = 1e-5)
expect_equal(run1(cos, c(0)), c(1), tolerance = 1e-5)
expect_equal(run1(tanh, c(0)), c(0), tolerance = 1e-5)

# Compound expressions
expect_equal(
  run1(function(x) (1 - exp(-x)) / (1 + exp(-x)), c(0)),
  c(0), tolerance = 1e-5
)
