#ifndef RJAX_H
#define RJAX_H

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <dlfcn.h>

/*
 * Opaque PJRT types.
 *
 * The real definitions live in pjrt_c_api.h from the XLA project.
 * We forward-declare them here so the package compiles without
 * XLA headers at build time. The actual structs are only touched
 * through the PJRT_Api function pointer table at runtime.
 */
typedef struct PJRT_Api PJRT_Api;
typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Buffer PJRT_Buffer;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Error PJRT_Error;

/* Global state: the loaded PJRT plugin */
extern const PJRT_Api *pjrt_api;
extern void *pjrt_plugin_handle;
extern PJRT_Client *pjrt_client;

/* ---- pjrt_plugin.c ---- */
SEXP rjax_load_plugin(SEXP path);
SEXP rjax_unload_plugin(void);

/* ---- pjrt_client.c ---- */
SEXP rjax_client_create(void);
SEXP rjax_client_destroy(SEXP client_ptr);
SEXP rjax_client_devices(SEXP client_ptr);

/* ---- pjrt_buffer.c ---- */
SEXP rjax_buffer_from_r(SEXP client_ptr, SEXP data, SEXP dims, SEXP dtype);
SEXP rjax_buffer_to_r(SEXP buffer_ptr);
SEXP rjax_buffer_destroy(SEXP buffer_ptr);

/* ---- pjrt_compile.c ---- */
SEXP rjax_compile(SEXP client_ptr, SEXP hlo_bytes);
SEXP rjax_execute(SEXP exec_ptr, SEXP input_buffers);
SEXP rjax_executable_destroy(SEXP exec_ptr);

/* Error handling helper */
#define RJAX_CHECK_API() do {                                 \
    if (!pjrt_api) Rf_error("PJRT plugin not loaded");       \
} while (0)

#define RJAX_CHECK_PTR(x, label) do {                         \
    if (R_ExternalPtrAddr(x) == NULL)                         \
        Rf_error(label " has been destroyed or is invalid");  \
} while (0)

#endif /* RJAX_H */
