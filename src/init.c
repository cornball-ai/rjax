#include "rjax.h"

static const R_CallMethodDef call_methods[] = {
    /* pjrt_plugin.c */
    {"rjax_load_plugin",        (DL_FUNC) &rjax_load_plugin,        1},
    {"rjax_unload_plugin",      (DL_FUNC) &rjax_unload_plugin,      0},

    /* pjrt_client.c */
    {"rjax_client_create",      (DL_FUNC) &rjax_client_create,      0},
    {"rjax_client_destroy",     (DL_FUNC) &rjax_client_destroy,     1},
    {"rjax_client_devices",     (DL_FUNC) &rjax_client_devices,     1},

    /* pjrt_buffer.c */
    {"rjax_buffer_from_r",      (DL_FUNC) &rjax_buffer_from_r,      4},
    {"rjax_buffer_to_r",        (DL_FUNC) &rjax_buffer_to_r,        1},
    {"rjax_buffer_destroy",     (DL_FUNC) &rjax_buffer_destroy,     1},

    /* pjrt_compile.c */
    {"rjax_compile",            (DL_FUNC) &rjax_compile,             2},
    {"rjax_execute",            (DL_FUNC) &rjax_execute,             2},
    {"rjax_executable_destroy", (DL_FUNC) &rjax_executable_destroy,  1},

    {NULL, NULL, 0}
};

void R_init_rjax(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
