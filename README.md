# rjax

R interface to XLA. Write normal R math, compile it to optimized machine code, run it on CPU or GPU, and differentiate it automatically.

No Python. No JAX. Just R and the XLA compiler.

> **Note:** This is an experimental project. If you want a more mature R+XLA stack with active development and broader feature coverage, check out [anvil](https://github.com/r-xla/anvil) from the r-xla org.

## Quick start

```r
library(rjax)

# Write a function using normal R math
tanh_custom <- function(x) {
  y <- exp(-2 * x)
  (1 - y) / (1 + y)
}

# JIT compile it
fast_tanh <- xla_jit(tanh_custom)
fast_tanh(1.0)
#> [1] 0.7615942

# Differentiate it
grad_tanh <- xla_grad(tanh_custom)
grad_tanh(1.0)
#> [1] 0.4199743

# Higher-order derivatives
xla_grad(xla_grad(function(x) x^3))(2.0)
#> [1] 12

# Vector inputs
xla_grad(function(w) sum(w^2))(c(1, 2, 3))
#> [1] 2 4 6
```

## GPU

```r
fast_tanh <- xla_jit(tanh_custom, backend = "gpu")
fast_tanh(1.0)

grad_tanh <- xla_grad(tanh_custom, backend = "gpu")
grad_tanh(1.0)
```

Requires the CUDA build of the XLA extension library. Set `XLA_VARIANT=cuda12` before `configure` to download the CUDA variant automatically.

## What works

**Operator overloading**: `+`, `-`, `*`, `/`, `^`, and math functions (`exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `abs`) on XLA computation graphs. Scalars auto-promote.

**JIT compilation** (`xla_jit`): traces an R function into an XLA computation, compiles it, caches by input shape. Subsequent calls skip compilation.

**Reverse-mode autodiff** (`xla_grad`): tape-based automatic differentiation. VJP rules for all supported operations. Supports scalar and vector inputs, multi-argument functions, and higher-order derivatives for polynomial operations.

**Matrix operations**: `xla_dot` (matrix multiply), `xla_transpose`, `xla_reshape` with gradient support.

**Backends**: CPU and CUDA GPU via the PJRT runtime.

## Installation

```r
# From GitHub
remotes::install_github("cornball-ai/rjax")
```

The `configure` script downloads a prebuilt XLA extension library (~500 MB) from [elixir-nx/xla](https://github.com/elixir-nx/xla/releases). Requires a C++ compiler with C++17 support.

For GPU support:

```bash
XLA_VARIANT=cuda12 R CMD INSTALL .
```

## How it works

rjax links directly to `libxla_extension.so` (the XLA compiler and runtime) via Rcpp. No intermediate Python layer.

- `xla_jit(f)` traces `f` by calling it with abstract `xla_op` values. Overloaded R operators build an XLA computation graph. The graph is compiled once, then executed on subsequent calls.
- `xla_grad(f)` traces the forward pass while recording a tape of operations, then walks the tape backward applying VJP (vector-Jacobian product) rules to construct a gradient computation. The gradient computation is itself an XLA graph, compiled and executed on the same backend.
- The returned gradient function is itself traceable, enabling `xla_grad(xla_grad(f))` for higher-order derivatives.

## Known limitations

- Higher-order derivatives work for polynomial operations (`x^n`, `+`, `-`, `*`, `/`). Operations whose VJP rules reference the forward output (`exp`, `tanh`, `sin`) don't yet support grad-of-grad. Fixing this requires moving from tape-based AD to a Jaxpr-style functional IR.
- XLA uses row-major layout. R matrices are column-major. Pass `byrow = TRUE` when creating matrices, or expect transposed results.
- No automatic broadcasting across different-rank tensors yet. Scalar-tensor broadcasting works.

## License

MIT
