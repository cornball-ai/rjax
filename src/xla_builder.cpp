#include "rjax_types.h"

// [[Rcpp::export]]
SEXP rjax_builder_new(std::string name) {
  auto builder = std::make_shared<xla::XlaBuilder>(name);
  return Rcpp::XPtr<XlaBuilderPtr>(new XlaBuilderPtr(builder), true);
}

// [[Rcpp::export]]
SEXP rjax_parameter(SEXP builder_xptr, int param_num,
                    std::string dtype, Rcpp::IntegerVector dims) {
  Rcpp::XPtr<XlaBuilderPtr> bptr(builder_xptr);
  auto& builder = **bptr;

  xla::PrimitiveType ptype = dtype_from_string(dtype);

  std::vector<int64_t> shape_dims(dims.size());
  for (int i = 0; i < dims.size(); i++) shape_dims[i] = dims[i];

  xla::Shape shape = xla::ShapeUtil::MakeShape(ptype, shape_dims);
  xla::XlaOp op = xla::Parameter(&builder, param_num, shape,
                                  "p" + std::to_string(param_num));

  auto op_ptr = std::make_shared<xla::XlaOp>(op);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(op_ptr), true);
}

// [[Rcpp::export]]
SEXP rjax_constant_r1(SEXP builder_xptr, Rcpp::NumericVector values,
                      std::string dtype) {
  Rcpp::XPtr<XlaBuilderPtr> bptr(builder_xptr);
  auto& builder = **bptr;

  xla::XlaOp op;
  if (dtype == "f32") {
    std::vector<float> fvals(values.size());
    for (int i = 0; i < values.size(); i++)
      fvals[i] = static_cast<float>(values[i]);
    op = xla::ConstantR1<float>(&builder, fvals);
  } else if (dtype == "f64") {
    std::vector<double> dvals(values.begin(), values.end());
    op = xla::ConstantR1<double>(&builder, dvals);
  } else if (dtype == "i32") {
    std::vector<int32_t> ivals(values.size());
    for (int i = 0; i < values.size(); i++)
      ivals[i] = static_cast<int32_t>(values[i]);
    op = xla::ConstantR1<int32_t>(&builder, ivals);
  } else {
    Rcpp::stop("Unsupported dtype for constant: " + dtype);
  }

  auto op_ptr = std::make_shared<xla::XlaOp>(op);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(op_ptr), true);
}

// [[Rcpp::export]]
SEXP rjax_constant_scalar(SEXP builder_xptr, double value, std::string dtype) {
  Rcpp::XPtr<XlaBuilderPtr> bptr(builder_xptr);
  auto& builder = **bptr;

  xla::XlaOp op;
  if (dtype == "f32") {
    op = xla::ConstantR0<float>(&builder, static_cast<float>(value));
  } else if (dtype == "f64") {
    op = xla::ConstantR0<double>(&builder, value);
  } else if (dtype == "i32") {
    op = xla::ConstantR0<int32_t>(&builder, static_cast<int32_t>(value));
  } else {
    Rcpp::stop("Unsupported dtype for scalar constant: " + dtype);
  }

  auto op_ptr = std::make_shared<xla::XlaOp>(op);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(op_ptr), true);
}

// ---- Arithmetic operations ----

// [[Rcpp::export]]
SEXP rjax_add(SEXP lhs_xptr, SEXP rhs_xptr) {
  Rcpp::XPtr<XlaOpPtr> lptr(lhs_xptr);
  Rcpp::XPtr<XlaOpPtr> rptr(rhs_xptr);
  auto op = xla::Add(**lptr, **rptr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_sub(SEXP lhs_xptr, SEXP rhs_xptr) {
  Rcpp::XPtr<XlaOpPtr> lptr(lhs_xptr);
  Rcpp::XPtr<XlaOpPtr> rptr(rhs_xptr);
  auto op = xla::Sub(**lptr, **rptr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_mul(SEXP lhs_xptr, SEXP rhs_xptr) {
  Rcpp::XPtr<XlaOpPtr> lptr(lhs_xptr);
  Rcpp::XPtr<XlaOpPtr> rptr(rhs_xptr);
  auto op = xla::Mul(**lptr, **rptr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_div(SEXP lhs_xptr, SEXP rhs_xptr) {
  Rcpp::XPtr<XlaOpPtr> lptr(lhs_xptr);
  Rcpp::XPtr<XlaOpPtr> rptr(rhs_xptr);
  auto op = xla::Div(**lptr, **rptr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_neg(SEXP op_xptr) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  auto op = xla::Neg(**optr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_abs(SEXP op_xptr) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  auto op = xla::Abs(**optr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_exp(SEXP op_xptr) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  auto op = xla::Exp(**optr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_log(SEXP op_xptr) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  auto op = xla::Log(**optr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_sqrt(SEXP op_xptr) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  auto op = xla::Sqrt(**optr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_tanh(SEXP op_xptr) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  auto op = xla::Tanh(**optr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// ---- Matrix operations ----

// [[Rcpp::export]]
SEXP rjax_dot(SEXP lhs_xptr, SEXP rhs_xptr) {
  Rcpp::XPtr<XlaOpPtr> lptr(lhs_xptr);
  Rcpp::XPtr<XlaOpPtr> rptr(rhs_xptr);
  auto op = xla::Dot(**lptr, **rptr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_transpose(SEXP op_xptr, Rcpp::IntegerVector permutation) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  std::vector<int64_t> perm(permutation.size());
  for (int i = 0; i < permutation.size(); i++) perm[i] = permutation[i];
  auto op = xla::Transpose(**optr, perm);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_reshape(SEXP op_xptr, Rcpp::IntegerVector new_dims) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  std::vector<int64_t> dims(new_dims.size());
  for (int i = 0; i < new_dims.size(); i++) dims[i] = new_dims[i];
  auto op = xla::Reshape(**optr, dims);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// ---- Build computation ----

// [[Rcpp::export]]
SEXP rjax_build(SEXP builder_xptr) {
  Rcpp::XPtr<XlaBuilderPtr> bptr(builder_xptr);
  auto computation = unwrap((*bptr)->Build());
  auto comp_ptr = std::make_shared<xla::XlaComputation>(std::move(computation));
  return Rcpp::XPtr<XlaComputationPtr>(new XlaComputationPtr(comp_ptr), true);
}

// Build with a specific root op
// [[Rcpp::export]]
SEXP rjax_build_with_root(SEXP builder_xptr, SEXP root_xptr) {
  Rcpp::XPtr<XlaBuilderPtr> bptr(builder_xptr);
  Rcpp::XPtr<XlaOpPtr> rptr(root_xptr);
  auto computation = unwrap((*bptr)->Build(**rptr));
  auto comp_ptr = std::make_shared<xla::XlaComputation>(std::move(computation));
  return Rcpp::XPtr<XlaComputationPtr>(new XlaComputationPtr(comp_ptr), true);
}
