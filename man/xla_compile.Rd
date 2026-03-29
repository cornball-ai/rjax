% tinyrox says don't edit this manually, but it can't stop you!
\name{xla_compile}
\alias{xla_compile}
\title{Compile an XLA computation}
\usage{
xla_compile(client, computation)
}
\arguments{
\item{client}{An \code{xla_client} object.}

\item{computation}{An \code{xla_computation} object from \code{xla_build}.}
}
\value{
An external pointer of class \code{xla_executable}.
}
\description{
Compiles an XlaComputation into an executable for the client's backend.
}
