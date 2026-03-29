#include "rjax_types.h"

// [[Rcpp::export]]
SEXP rjax_buffer_from_r(SEXP client_xptr, SEXP data_sexp,
                        SEXP dims_sexp, SEXP dtype_sexp) {
  Rcpp::XPtr<PjRtClientPtr> xptr(client_xptr);
  auto& client = **xptr;

  Rcpp::NumericVector data(data_sexp);
  Rcpp::IntegerVector dims(dims_sexp);
  std::string dtype = Rcpp::as<std::string>(dtype_sexp);

  xla::PrimitiveType ptype = dtype_from_string(dtype);

  std::vector<int64_t> shape_dims(dims.size());
  for (int i = 0; i < dims.size(); i++) {
    shape_dims[i] = dims[i];
  }

  size_t n = data.size();
  auto* device_mem = unwrap(client.addressable_devices()[0]->default_memory_space());
  auto semantics = xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;

  std::unique_ptr<xla::PjRtBuffer> buffer;

  if (ptype == xla::F32) {
    std::vector<float> fdata(n);
    for (size_t i = 0; i < n; i++) fdata[i] = static_cast<float>(data[i]);
    buffer = unwrap(client.BufferFromHostBuffer(
        fdata.data(), ptype, shape_dims, std::nullopt,
        semantics, nullptr, device_mem, nullptr));
  } else if (ptype == xla::F64) {
    buffer = unwrap(client.BufferFromHostBuffer(
        REAL(data_sexp), ptype, shape_dims, std::nullopt,
        semantics, nullptr, device_mem, nullptr));
  } else if (ptype == xla::S32) {
    std::vector<int32_t> idata(n);
    for (size_t i = 0; i < n; i++) idata[i] = static_cast<int32_t>(data[i]);
    buffer = unwrap(client.BufferFromHostBuffer(
        idata.data(), ptype, shape_dims, std::nullopt,
        semantics, nullptr, device_mem, nullptr));
  } else {
    Rcpp::stop("Unsupported dtype for buffer creation: " + dtype);
  }

  PjRtBufferPtr shared_buf(buffer.release());
  return Rcpp::XPtr<PjRtBufferPtr>(new PjRtBufferPtr(shared_buf), true);
}

// [[Rcpp::export]]
SEXP rjax_buffer_to_r(SEXP buffer_xptr) {
  Rcpp::XPtr<PjRtBufferPtr> xptr(buffer_xptr);
  auto& buffer = **xptr;

  auto literal = unwrap(buffer.ToLiteralSync());
  auto shape = buffer.on_device_shape();
  auto ptype = shape.element_type();
  auto dims = shape.dimensions();

  size_t n = literal->element_count();
  Rcpp::NumericVector result(n);

  if (ptype == xla::F32) {
    auto span = literal->data<float>();
    for (size_t i = 0; i < n; i++) result[i] = span[i];
  } else if (ptype == xla::F64) {
    auto span = literal->data<double>();
    for (size_t i = 0; i < n; i++) result[i] = span[i];
  } else if (ptype == xla::S32) {
    auto span = literal->data<int32_t>();
    for (size_t i = 0; i < n; i++) result[i] = span[i];
  } else {
    Rcpp::stop("Unsupported dtype for buffer-to-R conversion: " +
               dtype_to_string(ptype));
  }

  if (dims.size() > 1) {
    Rcpp::IntegerVector rdims(dims.size());
    for (size_t i = 0; i < dims.size(); i++) rdims[i] = dims[i];
    result.attr("dim") = rdims;
  }

  return Rcpp::wrap(result);
}

// [[Rcpp::export]]
SEXP rjax_buffer_shape(SEXP buffer_xptr) {
  Rcpp::XPtr<PjRtBufferPtr> xptr(buffer_xptr);
  auto& buffer = **xptr;

  auto shape = buffer.on_device_shape();
  auto dims = shape.dimensions();

  Rcpp::IntegerVector rdims(dims.size());
  for (size_t i = 0; i < dims.size(); i++) rdims[i] = dims[i];

  return Rcpp::List::create(
      Rcpp::Named("dims") = rdims,
      Rcpp::Named("dtype") = dtype_to_string(shape.element_type()));
}
