% tinyrox says don't edit this manually, but it can't stop you!
\name{xla_buffer}
\alias{xla_buffer}
\title{Transfer data to an XLA device buffer}
\usage{
xla_buffer(client, data, dims = length(data), dtype = "f32")
}
\arguments{
\item{client}{An \code{xla_client} object.}

\item{data}{A numeric or integer vector.}

\item{dims}{Integer vector of dimensions. Defaults to \code{length(data)}
(a 1-D tensor).}

\item{dtype}{Character scalar: \code{"f32"}, \code{"f64"}, \code{"i32"}.
Defaults to \code{"f32"}.}
}
\value{
An external pointer of class \code{xla_buffer}.
}
\description{
Copies an R numeric vector to device memory managed by the PJRT client.
}
