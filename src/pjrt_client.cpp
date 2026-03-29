#include "rjax_types.h"

// [[Rcpp::export]]
SEXP rjax_client_create() {
  xla::CpuClientOptions options;
  options.cpu_device_count = 1;

  auto client = unwrap(xla::GetPjRtCpuClient(options));
  PjRtClientPtr shared_client(client.release());
  return Rcpp::XPtr<PjRtClientPtr>(new PjRtClientPtr(shared_client), true);
}

// [[Rcpp::export]]
SEXP rjax_client_platform(SEXP client_xptr) {
  Rcpp::XPtr<PjRtClientPtr> xptr(client_xptr);
  auto& client = **xptr;
  return Rcpp::wrap(std::string(client.platform_name()));
}

// [[Rcpp::export]]
SEXP rjax_client_devices(SEXP client_xptr) {
  Rcpp::XPtr<PjRtClientPtr> xptr(client_xptr);
  auto& client = **xptr;
  auto devices = client.addressable_devices();

  Rcpp::CharacterVector result(devices.size());
  for (size_t i = 0; i < devices.size(); i++) {
    result[i] = std::string(devices[i]->ToString());
  }
  return Rcpp::wrap(result);
}
