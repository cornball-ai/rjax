% tinyrox says don't edit this manually, but it can't stop you!
\name{xla_client}
\alias{xla_client}
\title{Create an XLA client}
\usage{
xla_client()
}
\value{
An external pointer of class \code{xla_client}.
}
\description{
Creates a PJRT client for the loaded plugin backend.
The PJRT plugin must be loaded first (either via PJRT_PLUGIN_PATH
environment variable or by calling the plugin loader directly).
}
