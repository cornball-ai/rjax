#' @useDynLib rjax, .registration = TRUE

.onLoad <- function(libname, pkgname) {
  plugin_path <- Sys.getenv("PJRT_PLUGIN_PATH", unset = "")

  if (nzchar(plugin_path)) {
    tryCatch(
      .Call(rjax_load_plugin, plugin_path),
      error = function(e) {
        warning("rjax: failed to load PJRT plugin: ", e$message, call. = FALSE)
      }
    )
  }
}

.onUnload <- function(libpath) {
  .Call(rjax_unload_plugin)
}
