#include "rjax_types.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"

// [[Rcpp::export]]
SEXP rjax_client_create(std::string backend) {
  std::unique_ptr<xla::PjRtClient> client;

  if (backend == "cpu") {
    xla::CpuClientOptions options;
    options.cpu_device_count = 1;
    client = unwrap(xla::GetPjRtCpuClient(options));
  } else if (backend == "gpu" || backend == "cuda") {
    xla::GpuClientOptions options;
    client = unwrap(xla::GetStreamExecutorGpuClient(options));
  } else {
    Rcpp::stop("Unknown backend: " + backend + ". Use 'cpu' or 'gpu'.");
  }

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
