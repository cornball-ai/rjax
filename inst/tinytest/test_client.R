# Test client creation
client <- xla_client()
expect_true(inherits(client, "xla_client"))

devices <- xla_devices(client)
expect_true(is.character(devices))
expect_true(length(devices) >= 1)

# Test buffer round-trip
buf <- xla_buffer(client, c(1, 2, 3), dtype = "f32")
expect_true(inherits(buf, "xla_buffer"))

vals <- as.array(buf)
expect_equal(vals, c(1, 2, 3), tolerance = 1e-6)

# Test f64 buffer
buf64 <- xla_buffer(client, c(1.5, 2.5), dtype = "f64")
vals64 <- as.array(buf64)
expect_equal(vals64, c(1.5, 2.5))

# Test buffer shape
info <- rjax:::rjax_buffer_shape(buf)
expect_equal(info$dtype, "f32")
expect_equal(info$dims, 3L)

# Test builder + compile + execute: a + b
builder <- xla_builder("add")
p0 <- xla_parameter(builder, 0L, "f32", 3L)
p1 <- xla_parameter(builder, 1L, "f32", 3L)
xla_add(p0, p1)
comp <- xla_build(builder)
expect_true(inherits(comp, "xla_computation"))

exec <- xla_compile(client, comp)
expect_true(inherits(exec, "xla_executable"))

a <- xla_buffer(client, c(1, 2, 3))
b <- xla_buffer(client, c(4, 5, 6))
result <- xla_execute(exec, list(a, b))
expect_equal(length(result), 1)
expect_true(inherits(result[[1]], "xla_buffer"))
expect_equal(as.array(result[[1]]), c(5, 7, 9), tolerance = 1e-6)

# Test multiply
builder2 <- xla_builder("mul")
p0 <- xla_parameter(builder2, 0L, "f32", 3L)
p1 <- xla_parameter(builder2, 1L, "f32", 3L)
xla_mul(p0, p1)
comp2 <- xla_build(builder2)
exec2 <- xla_compile(client, comp2)
result2 <- xla_execute(exec2, list(a, b))
expect_equal(as.array(result2[[1]]), c(4, 10, 18), tolerance = 1e-6)

# Test exp
builder3 <- xla_builder("exp")
p0 <- xla_parameter(builder3, 0L, "f32", 3L)
xla_exp(p0)
comp3 <- xla_build(builder3)
exec3 <- xla_compile(client, comp3)
zeros <- xla_buffer(client, c(0, 0, 0))
result3 <- xla_execute(exec3, list(zeros))
expect_equal(as.array(result3[[1]]), c(1, 1, 1), tolerance = 1e-6)

# Test constant
builder4 <- xla_builder("const_add")
p0 <- xla_parameter(builder4, 0L, "f32", 3L)
c1 <- xla_constant(builder4, c(10, 20, 30), "f32")
xla_add(p0, c1)
comp4 <- xla_build(builder4)
exec4 <- xla_compile(client, comp4)
result4 <- xla_execute(exec4, list(a))
expect_equal(as.array(result4[[1]]), c(11, 22, 33), tolerance = 1e-6)
