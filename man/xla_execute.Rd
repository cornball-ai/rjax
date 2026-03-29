% tinyrox says don't edit this manually, but it can't stop you!
\name{xla_execute}
\alias{xla_execute}
\title{Execute a compiled XLA program}
\usage{
xla_execute(executable, inputs = list())
}
\arguments{
\item{executable}{An \code{xla_executable} object.}

\item{inputs}{A list of \code{xla_buffer} objects.}
}
\value{
A list of \code{xla_buffer} objects (one per output).
}
\description{
Runs a compiled executable with the given input buffers.
}
