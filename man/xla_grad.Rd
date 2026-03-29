% tinyrox says don't edit this manually, but it can't stop you!
\name{xla_grad}
\alias{xla_grad}
\title{Create a gradient function via reverse-mode autodiff}
\usage{
xla_grad(f, dtype = "f32", argnum = 0L, backend = "cpu")
}
\arguments{
\item{f}{An R function using supported XLA operations.}

\item{dtype}{Default dtype: "f32" or "f64".}

\item{argnum}{Which argument to differentiate with respect to (0-indexed).}
}
\value{
A function that computes gradients.
}
\description{
Returns a new function that computes the gradient of \code{f} with respect
to its first argument. The returned function is itself traceable, so
\code{xla_grad(xla_grad(f))} computes second derivatives.
}
