% tinyrox says don't edit this manually, but it can't stop you!
\name{xla_constant}
\alias{xla_constant}
\title{Create a constant vector}
\usage{
xla_constant(builder, values, dtype = "f32")
}
\arguments{
\item{builder}{An \code{xla_builder} object.}

\item{values}{Numeric vector of values.}

\item{dtype}{Character: \code{"f32"}, \code{"f64"}, \code{"i32"}.}
}
\value{
An \code{xla_op} external pointer.
}
\description{
Create a constant vector
}
