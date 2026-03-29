#include "rjax.h"

/* Global state */
const PJRT_Api *pjrt_api = NULL;
void *pjrt_plugin_handle = NULL;
PJRT_Client *pjrt_client = NULL;

/*
 * Load a PJRT plugin shared library and retrieve the API table.
 *
 * The plugin exports a single entry point: GetPjrtApi()
 * which returns a PJRT_Api* (struct of function pointers).
 */
SEXP rjax_load_plugin(SEXP path)
{
    const char *plugin_path = CHAR(STRING_ELT(path, 0));

    if (pjrt_plugin_handle) {
        Rf_warning("PJRT plugin already loaded, unloading first");
        rjax_unload_plugin();
    }

    pjrt_plugin_handle = dlopen(plugin_path, RTLD_LAZY);
    if (!pjrt_plugin_handle) {
        Rf_error("Failed to load PJRT plugin '%s': %s",
                 plugin_path, dlerror());
    }

    /* The PJRT plugin entry point */
    typedef const PJRT_Api *(*GetPjrtApiFn)(void);
    GetPjrtApiFn get_api = (GetPjrtApiFn) dlsym(pjrt_plugin_handle, "GetPjrtApi");

    if (!get_api) {
        dlclose(pjrt_plugin_handle);
        pjrt_plugin_handle = NULL;
        Rf_error("PJRT plugin '%s' has no GetPjrtApi symbol: %s",
                 plugin_path, dlerror());
    }

    pjrt_api = get_api();
    if (!pjrt_api) {
        dlclose(pjrt_plugin_handle);
        pjrt_plugin_handle = NULL;
        Rf_error("GetPjrtApi() returned NULL");
    }

    return R_NilValue;
}

/*
 * Unload the PJRT plugin and reset global state.
 */
SEXP rjax_unload_plugin(void)
{
    /* TODO: destroy client and all outstanding buffers/executables first */
    pjrt_api = NULL;
    pjrt_client = NULL;

    if (pjrt_plugin_handle) {
        dlclose(pjrt_plugin_handle);
        pjrt_plugin_handle = NULL;
    }

    return R_NilValue;
}
