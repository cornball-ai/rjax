#include "rjax.h"

static void buffer_release(SEXP ptr)
{
    PJRT_Buffer *buf = (PJRT_Buffer *) R_ExternalPtrAddr(ptr);
    if (buf && pjrt_api) {
        /* TODO: call pjrt_api->PJRT_Buffer_Destroy() */
    }
    R_ClearExternalPtr(ptr);
}

/*
 * Transfer an R numeric vector to device memory.
 *
 * @param client_ptr  External pointer to PJRT_Client
 * @param data        R numeric vector (doubles)
 * @param dims        Integer vector of dimensions
 * @param dtype       Character scalar: "f32", "f64", "i32", etc.
 *
 * Returns an external pointer of class "xla_buffer".
 */
SEXP rjax_buffer_from_r(SEXP client_ptr, SEXP data, SEXP dims, SEXP dtype)
{
    RJAX_CHECK_API();
    RJAX_CHECK_PTR(client_ptr, "xla_client");

    /*
     * TODO:
     * 1. Convert R data to the requested dtype (float32, etc.)
     * 2. Fill PJRT_Client_BufferFromHostBuffer_Args
     * 3. Call pjrt_api->PJRT_Client_BufferFromHostBuffer()
     * 4. Wrap resulting PJRT_Buffer* in an external pointer
     */
    PJRT_Buffer *buf = NULL;

    SEXP ptr = PROTECT(R_MakeExternalPtr(buf, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, buffer_release, TRUE);
    setAttrib(ptr, R_ClassSymbol, mkString("xla_buffer"));
    UNPROTECT(1);
    return ptr;
}

/*
 * Copy device buffer contents back to an R vector.
 */
SEXP rjax_buffer_to_r(SEXP buffer_ptr)
{
    RJAX_CHECK_API();
    RJAX_CHECK_PTR(buffer_ptr, "xla_buffer");

    /*
     * TODO:
     * 1. Call pjrt_api->PJRT_Buffer_ToHostBuffer()
     * 2. Determine shape/dtype from buffer metadata
     * 3. Allocate R vector of appropriate type
     * 4. Copy data and set dim attribute
     */

    /* Stub: return NA */
    return ScalarReal(NA_REAL);
}

SEXP rjax_buffer_destroy(SEXP buffer_ptr)
{
    buffer_release(buffer_ptr);
    return R_NilValue;
}
