#' @useDynLib rjax, .registration = TRUE
#' @importFrom Rcpp evalCpp

.onLoad <- function(libname, pkgname) {
  # XLA library is linked at compile time, no runtime loading needed
}
