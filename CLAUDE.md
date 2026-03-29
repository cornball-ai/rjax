# rjax

R interface to XLA via the PJRT C API. No Python dependency.

## Architecture

Two planned layers:

1. **PJRT runtime (pure C)** - Plugin loading, client/buffer/executable lifecycle, execution.
   Uses `dlopen` + PJRT C API function pointer table. All R-to-C via `.Call()`.

2. **HLO builder (C++, future)** - Programmatic HLO construction via XlaBuilder.
   Will need C++ and linking against XLA headers/libs. Not yet implemented.

## Build/test

```bash
r -e 'tinyrox::document(); tinypkgr::install(); tinytest::test_package("rjax")'
r -e 'tinypkgr::check()'
```

## Key types

All XLA objects are R external pointers with S3 classes:

- `xla_client` - wraps `PJRT_Client*`
- `xla_buffer` - wraps `PJRT_Buffer*`
- `xla_executable` - wraps `PJRT_LoadedExecutable*`

## PJRT plugin

The package loads a PJRT plugin (`.so`) at runtime. Set `PJRT_PLUGIN_PATH` env var
to point at the plugin, or place it in `inst/lib/`. The `configure` script can
download prebuilt plugins from elixir-nx/xla releases.

## Dependencies

None in Imports. Pure base R + C.
