#include "rjax.h"

static void client_release(SEXP ptr)
{
    PJRT_Client *client = (PJRT_Client *) R_ExternalPtrAddr(ptr);
    if (client && pjrt_api) {
        /* TODO: call pjrt_api->PJRT_Client_Destroy() */
    }
    R_ClearExternalPtr(ptr);
}

/*
 * Create a PJRT client for the loaded plugin's default backend.
 * Returns an external pointer of class "xla_client".
 */
SEXP rjax_client_create(void)
{
    RJAX_CHECK_API();

    /*
     * TODO: fill in PJRT_Client_Create_Args and call
     * pjrt_api->PJRT_Client_Create(&args)
     */
    PJRT_Client *client = NULL;

    SEXP ptr = PROTECT(R_MakeExternalPtr(client, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, client_release, TRUE);
    setAttrib(ptr, R_ClassSymbol, mkString("xla_client"));
    UNPROTECT(1);
    return ptr;
}

SEXP rjax_client_destroy(SEXP client_ptr)
{
    client_release(client_ptr);
    return R_NilValue;
}

/*
 * List devices available on the client.
 * Returns a character vector of device descriptions.
 */
SEXP rjax_client_devices(SEXP client_ptr)
{
    RJAX_CHECK_API();
    RJAX_CHECK_PTR(client_ptr, "xla_client");

    /*
     * TODO: call pjrt_api->PJRT_Client_Devices()
     * iterate over devices, extract descriptions
     */

    /* Stub: return empty character vector */
    return allocVector(STRSXP, 0);
}
