# rjax

R interface to XLA. No Python dependency.

## Architecture

```
R layer (xla_client, xla_buffer, xla_builder, xla_compile, xla_execute)
  -> Rcpp C++ layer (src/*.cpp)
    -> libxla_extension.so (elixir-nx prebuilt, downloaded by configure)
```

Uses the XLA C++ API directly: PjRtClient, PjRtBuffer, XlaBuilder.

## Build/test

```bash
r -e 'Rscript -e "Rcpp::compileAttributes()"; tinyrox::document(); tinypkgr::install()'
r -e 'tinytest::test_package("rjax")'
```

After changing C++ code, regenerate Rcpp exports before building:
```bash
Rscript -e 'Rcpp::compileAttributes()'
```

## R/C++ header conflict

R defines macros (`Memcpy`, `Free`, `length`, etc.) that collide with XLA headers.
`rjax_types.h` handles this: includes Rcpp first, `#undef`s the offenders, then
includes XLA headers. All `.cpp` files must include `rjax_types.h` instead of
including Rcpp and XLA separately.

## Key types

All XLA objects are Rcpp::XPtr wrapping shared_ptr:

- `xla_client` - wraps `shared_ptr<PjRtClient>`
- `xla_buffer` - wraps `shared_ptr<PjRtBuffer>`
- `xla_executable` - wraps `shared_ptr<PjRtLoadedExecutable>`
- `xla_builder` - wraps `shared_ptr<XlaBuilder>`
- `xla_op` - wraps `shared_ptr<XlaOp>`
- `xla_computation` - wraps `shared_ptr<XlaComputation>`

## XLA extension

configure downloads `libxla_extension.so` + headers from elixir-nx/xla releases.
Stored in `inst/xla/` (gitignored, not shipped in package tarball).
Set `XLA_EXTENSION_DIR` env var to use a different location.

## Dependencies

Imports: Rcpp
LinkingTo: Rcpp
Runtime: libxla_extension.so (linked at compile time, rpath'd)
