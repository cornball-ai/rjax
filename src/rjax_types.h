#ifndef RJAX_TYPES_H
#define RJAX_TYPES_H

// R headers define macros (Memcpy, Free, length, etc.) that collide with
// XLA/abseil/stream_executor. Include Rcpp first, then undef the offenders,
// then include XLA.
#include <Rcpp.h>

// Undefine R macros that clash with XLA headers
#undef Memcpy
#undef Free
#undef length
#undef Realloc
#undef Calloc
#undef R_Calloc
#undef R_Free
#undef R_Realloc
#undef TRUE
#undef FALSE

#include <memory>
#include <string>
#include <vector>

#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"

// Restore R's TRUE/FALSE (Rboolean enum values)
// R internally uses Rboolean enum {FALSE=0, TRUE=1} but also
// defines TRUE/FALSE as macros. XLA doesn't use these names,
// so restoring them is safe.
#define TRUE ((Rboolean) 1)
#define FALSE ((Rboolean) 0)

// Shared pointer types for R external pointers
using PjRtClientPtr = std::shared_ptr<xla::PjRtClient>;
using PjRtBufferPtr = std::shared_ptr<xla::PjRtBuffer>;
using PjRtExecPtr = std::shared_ptr<xla::PjRtLoadedExecutable>;
using XlaBuilderPtr = std::shared_ptr<xla::XlaBuilder>;
using XlaComputationPtr = std::shared_ptr<xla::XlaComputation>;
using XlaOpPtr = std::shared_ptr<xla::XlaOp>;

// Unwrap StatusOr, throwing R errors on failure
template <typename T>
T unwrap(absl::StatusOr<T> status_or) {
  if (!status_or.ok()) {
    Rcpp::stop(std::string(status_or.status().message()));
  }
  return std::move(*status_or);
}

inline void check_status(absl::Status status) {
  if (!status.ok()) {
    Rcpp::stop(std::string(status.message()));
  }
}

// Map R dtype string to XLA PrimitiveType
inline xla::PrimitiveType dtype_from_string(const std::string& dtype) {
  if (dtype == "f32") return xla::F32;
  if (dtype == "f64") return xla::F64;
  if (dtype == "i32") return xla::S32;
  if (dtype == "i64") return xla::S64;
  if (dtype == "bool") return xla::PRED;
  Rcpp::stop("Unknown dtype: " + dtype);
  return xla::F32;
}

inline std::string dtype_to_string(xla::PrimitiveType type) {
  switch (type) {
    case xla::F32: return "f32";
    case xla::F64: return "f64";
    case xla::S32: return "i32";
    case xla::S64: return "i64";
    case xla::PRED: return "bool";
    default: return "unknown";
  }
}

#endif // RJAX_TYPES_H
