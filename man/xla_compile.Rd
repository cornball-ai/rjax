% tinyrox says don't edit this manually, but it can't stop you!
\name{xla_compile}
\alias{xla_compile}
\title{Compile an HLO module}
\usage{
xla_compile(client, hlo_bytes)
}
\arguments{
\item{client}{An \code{xla_client} object.}

\item{hlo_bytes}{A raw vector containing a serialized HLO module proto.}
}
\value{
An external pointer of class \code{xla_executable}.
}
\description{
Takes serialized HLO module bytes (protobuf) and compiles them
into an executable for the client's backend.
}
