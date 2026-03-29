#include "rjax.h"

static void executable_release(SEXP ptr)
{
    PJRT_LoadedExecutable *exec =
        (PJRT_LoadedExecutable *) R_ExternalPtrAddr(ptr);
    if (exec && pjrt_api) {
        /* TODO: call pjrt_api->PJRT_LoadedExecutable_Delete() */
    }
    R_ClearExternalPtr(ptr);
}

/*
 * Compile serialized HLO bytes into an executable.
 *
 * @param client_ptr  External pointer to PJRT_Client
 * @param hlo_bytes   Raw vector containing serialized HLO module proto
 *
 * Returns an external pointer of class "xla_executable".
 */
SEXP rjax_compile(SEXP client_ptr, SEXP hlo_bytes)
{
    RJAX_CHECK_API();
    RJAX_CHECK_PTR(client_ptr, "xla_client");

    if (TYPEOF(hlo_bytes) != RAWSXP) {
        Rf_error("hlo_bytes must be a raw vector");
    }

    /*
     * TODO:
     * 1. Fill PJRT_Program with RAW(hlo_bytes) and LENGTH(hlo_bytes)
     * 2. Fill PJRT_Client_Compile_Args
     * 3. Call pjrt_api->PJRT_Client_Compile()
     * 4. Wrap PJRT_LoadedExecutable* in external pointer
     */
    PJRT_LoadedExecutable *exec = NULL;

    SEXP ptr = PROTECT(R_MakeExternalPtr(exec, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, executable_release, TRUE);
    setAttrib(ptr, R_ClassSymbol, mkString("xla_executable"));
    UNPROTECT(1);
    return ptr;
}

/*
 * Execute a compiled program.
 *
 * @param exec_ptr       External pointer to PJRT_LoadedExecutable
 * @param input_buffers  List of xla_buffer external pointers
 *
 * Returns a list of xla_buffer external pointers (one per output).
 */
SEXP rjax_execute(SEXP exec_ptr, SEXP input_buffers)
{
    RJAX_CHECK_API();
    RJAX_CHECK_PTR(exec_ptr, "xla_executable");

    if (TYPEOF(input_buffers) != VECSXP) {
        Rf_error("input_buffers must be a list of xla_buffer objects");
    }

    /*
     * TODO:
     * 1. Extract PJRT_Buffer* from each input list element
     * 2. Fill PJRT_LoadedExecutable_Execute_Args
     * 3. Call pjrt_api->PJRT_LoadedExecutable_Execute()
     * 4. Wrap output PJRT_Buffer* pointers in external pointers
     * 5. Return as list
     */

    /* Stub: return empty list */
    return allocVector(VECSXP, 0);
}

SEXP rjax_executable_destroy(SEXP exec_ptr)
{
    executable_release(exec_ptr);
    return R_NilValue;
}
