% tinyrox says don't edit this manually, but it can't stop you!
\name{xla_build}
\alias{xla_build}
\title{Build a computation from the builder}
\usage{
xla_build(builder, root = NULL)
}
\arguments{
\item{builder}{An \code{xla_builder} object.}

\item{root}{Optional \code{xla_op} to use as the root. If NULL, uses
the last operation.}
}
\value{
An \code{xla_computation} external pointer.
}
\description{
Builds the XLA computation graph. The last operation added becomes the
root (output) of the computation, unless \code{root} is specified.
}
