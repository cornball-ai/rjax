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

// ---- Additional math ops ----

// [[Rcpp::export]]
SEXP rjax_pow(SEXP lhs_xptr, SEXP rhs_xptr) {
  Rcpp::XPtr<XlaOpPtr> lptr(lhs_xptr);
  Rcpp::XPtr<XlaOpPtr> rptr(rhs_xptr);
  auto op = xla::Pow(**lptr, **rptr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_sin(SEXP op_xptr) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  auto op = xla::Sin(**optr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_cos(SEXP op_xptr) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  auto op = xla::Cos(**optr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_sign(SEXP op_xptr) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  auto op = xla::Sign(**optr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_max(SEXP lhs_xptr, SEXP rhs_xptr) {
  Rcpp::XPtr<XlaOpPtr> lptr(lhs_xptr);
  Rcpp::XPtr<XlaOpPtr> rptr(rhs_xptr);
  auto op = xla::Max(**lptr, **rptr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_min(SEXP lhs_xptr, SEXP rhs_xptr) {
  Rcpp::XPtr<XlaOpPtr> lptr(lhs_xptr);
  Rcpp::XPtr<XlaOpPtr> rptr(rhs_xptr);
  auto op = xla::Min(**lptr, **rptr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_clamp(SEXP min_xptr, SEXP op_xptr, SEXP max_xptr) {
  Rcpp::XPtr<XlaOpPtr> mn(min_xptr);
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  Rcpp::XPtr<XlaOpPtr> mx(max_xptr);
  auto op = xla::Clamp(**mn, **optr, **mx);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// [[Rcpp::export]]
SEXP rjax_select(SEXP pred_xptr, SEXP on_true_xptr, SEXP on_false_xptr) {
  Rcpp::XPtr<XlaOpPtr> pptr(pred_xptr);
  Rcpp::XPtr<XlaOpPtr> tptr(on_true_xptr);
  Rcpp::XPtr<XlaOpPtr> fptr(on_false_xptr);
  auto op = xla::Select(**pptr, **tptr, **fptr);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// ---- Shape and broadcast ----

// [[Rcpp::export]]
SEXP rjax_op_shape(SEXP builder_xptr, SEXP op_xptr) {
  Rcpp::XPtr<XlaBuilderPtr> bptr(builder_xptr);
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  auto shape = unwrap((*bptr)->GetShape(**optr));
  auto dims = shape.dimensions();

  Rcpp::IntegerVector rdims(dims.size());
  for (size_t i = 0; i < dims.size(); i++) rdims[i] = dims[i];

  return Rcpp::List::create(
      Rcpp::Named("dims") = rdims,
      Rcpp::Named("dtype") = dtype_to_string(shape.element_type()));
}

// [[Rcpp::export]]
SEXP rjax_broadcast_in_dim(SEXP op_xptr, SEXP out_dims_sexp,
                           SEXP broadcast_dims_sexp) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  Rcpp::IntegerVector out_dims_r(out_dims_sexp);
  Rcpp::IntegerVector bc_dims_r(broadcast_dims_sexp);

  std::vector<int64_t> out_dims(out_dims_r.size());
  for (int i = 0; i < out_dims_r.size(); i++) out_dims[i] = out_dims_r[i];

  std::vector<int64_t> bc_dims(bc_dims_r.size());
  for (int i = 0; i < bc_dims_r.size(); i++) bc_dims[i] = bc_dims_r[i];

  auto op = xla::BroadcastInDim(**optr, out_dims, bc_dims);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// Reduce sum: builds the internal add sub-computation
// [[Rcpp::export]]
SEXP rjax_reduce_sum(SEXP op_xptr, SEXP dims_sexp, SEXP builder_xptr) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  Rcpp::XPtr<XlaBuilderPtr> bptr(builder_xptr);
  Rcpp::IntegerVector dims_r(dims_sexp);

  // Get the element type of the operand
  auto shape = unwrap((*bptr)->GetShape(**optr));
  auto ptype = shape.element_type();

  // Build the add sub-computation
  xla::XlaBuilder sub_builder("add");
  auto sub_shape = xla::ShapeUtil::MakeShape(ptype, {});
  auto lhs = xla::Parameter(&sub_builder, 0, sub_shape, "lhs");
  auto rhs = xla::Parameter(&sub_builder, 1, sub_shape, "rhs");
  xla::Add(lhs, rhs);
  auto add_comp = unwrap(sub_builder.Build());

  // Zero init value
  xla::XlaOp init;
  if (ptype == xla::F32) {
    init = xla::ConstantR0<float>(&(**bptr), 0.0f);
  } else if (ptype == xla::F64) {
    init = xla::ConstantR0<double>(&(**bptr), 0.0);
  } else if (ptype == xla::S32) {
    init = xla::ConstantR0<int32_t>(&(**bptr), 0);
  } else {
    Rcpp::stop("Unsupported dtype for reduce_sum");
  }

  std::vector<int64_t> reduce_dims(dims_r.size());
  for (int i = 0; i < dims_r.size(); i++) reduce_dims[i] = dims_r[i];

  auto op = xla::Reduce(**optr, init, add_comp, reduce_dims);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// Scalar broadcast: create a scalar and broadcast to given shape
// [[Rcpp::export]]
SEXP rjax_constant_broadcast(SEXP builder_xptr, double value,
                             std::string dtype, SEXP dims_sexp) {
  Rcpp::XPtr<XlaBuilderPtr> bptr(builder_xptr);
  Rcpp::IntegerVector dims_r(dims_sexp);

  auto ptype = dtype_from_string(dtype);
  xla::XlaOp scalar;
  if (ptype == xla::F32) {
    scalar = xla::ConstantR0<float>(&(**bptr), static_cast<float>(value));
  } else if (ptype == xla::F64) {
    scalar = xla::ConstantR0<double>(&(**bptr), value);
  } else if (ptype == xla::S32) {
    scalar = xla::ConstantR0<int32_t>(&(**bptr), static_cast<int32_t>(value));
  } else {
    Rcpp::stop("Unsupported dtype for constant_broadcast");
  }

  std::vector<int64_t> out_dims(dims_r.size());
  for (int i = 0; i < dims_r.size(); i++) out_dims[i] = dims_r[i];

  // Broadcast scalar to shape
  auto op = xla::Broadcast(scalar, out_dims);
  return Rcpp::XPtr<XlaOpPtr>(new XlaOpPtr(std::make_shared<xla::XlaOp>(op)), true);
}

// Get the builder from an XlaOp (they're linked internally)
// [[Rcpp::export]]
SEXP rjax_op_builder(SEXP op_xptr) {
  Rcpp::XPtr<XlaOpPtr> optr(op_xptr);
  auto* builder = (**optr).builder();
  // Wrap as non-owning shared_ptr (builder is owned by the original XlaBuilderPtr)
  auto bptr = std::shared_ptr<xla::XlaBuilder>(
      std::shared_ptr<xla::XlaBuilder>{}, builder);
  return Rcpp::XPtr<XlaBuilderPtr>(new XlaBuilderPtr(bptr), true);
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
