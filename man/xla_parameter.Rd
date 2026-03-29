% tinyrox says don't edit this manually, but it can't stop you!
\name{xla_parameter}
\alias{xla_parameter}
\title{Add a parameter to the computation}
\usage{
xla_parameter(builder, param_num, dtype = "f32", dims = integer(0))
}
\arguments{
\item{builder}{An \code{xla_builder} object.}

\item{param_num}{Zero-based parameter index.}

\item{dtype}{Character: \code{"f32"}, \code{"f64"}, \code{"i32"}.}

\item{dims}{Integer vector of dimensions.}
}
\value{
An \code{xla_op} external pointer.
}
\description{
Add a parameter to the computation
}
