#include "rjax_types.h"

// [[Rcpp::export]]
SEXP rjax_compile(SEXP client_xptr, SEXP computation_xptr) {
  Rcpp::XPtr<PjRtClientPtr> cptr(client_xptr);
  Rcpp::XPtr<XlaComputationPtr> comp_ptr(computation_xptr);
  auto& client = **cptr;
  auto& computation = **comp_ptr;

  xla::CompileOptions options;
  auto executable = unwrap(client.CompileAndLoad(computation, options));

  PjRtExecPtr shared_exec(executable.release());
  return Rcpp::XPtr<PjRtExecPtr>(new PjRtExecPtr(shared_exec), true);
}

// [[Rcpp::export]]
SEXP rjax_execute(SEXP exec_xptr, SEXP input_buffers_sexp) {
  Rcpp::XPtr<PjRtExecPtr> eptr(exec_xptr);
  auto& executable = **eptr;

  Rcpp::List input_buffers(input_buffers_sexp);

  std::vector<xla::PjRtBuffer*> args;
  for (int i = 0; i < input_buffers.size(); i++) {
    SEXP buf_sexp = input_buffers[i];
    Rcpp::XPtr<PjRtBufferPtr> bptr(buf_sexp);
    args.push_back((*bptr).get());
  }

  xla::ExecuteOptions options;
  std::vector<std::vector<xla::PjRtBuffer*>> arg_handles = {args};

  auto results = unwrap(executable.Execute(arg_handles, options));

  Rcpp::List output(results[0].size());
  for (size_t i = 0; i < results[0].size(); i++) {
    PjRtBufferPtr shared_buf(results[0][i].release());
    output[i] = Rcpp::XPtr<PjRtBufferPtr>(new PjRtBufferPtr(shared_buf), true);
  }

  return Rcpp::wrap(output);
}
