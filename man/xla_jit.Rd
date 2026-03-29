% tinyrox says don't edit this manually, but it can't stop you!
\name{xla_jit}
\alias{xla_jit}
\title{JIT-compile an R function for XLA execution}
\usage{
xla_jit(f, dtype = "f32", backend = "cpu")
}
\arguments{
\item{f}{An R function using arithmetic ops (+, -, *, /, ^),
math functions (exp, log, sqrt, sin, cos, tanh), and sum().}

\item{dtype}{Default dtype for inputs: "f32" or "f64".}
}
\value{
A function that accepts numeric inputs and returns numeric output.
}
\description{
Returns a new function that traces the input function on first call,
compiles it with XLA, and caches the result. Subsequent calls with
the same input shapes reuse the cached compiled executable.
}
