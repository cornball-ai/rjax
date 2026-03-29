# Basic structure tests that don't require a PJRT plugin

expect_true(is.function(xla_client))
expect_true(is.function(xla_devices))
expect_true(is.function(xla_buffer))
expect_true(is.function(xla_compile))
expect_true(is.function(xla_execute))

# Tests that need a live plugin
if (at_home() && nzchar(Sys.getenv("PJRT_PLUGIN_PATH"))) {
  client <- xla_client()
  expect_true(inherits(client, "xla_client"))

  devices <- xla_devices(client)
  expect_true(is.character(devices))
}
