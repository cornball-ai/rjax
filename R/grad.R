#' Create a gradient function via reverse-mode autodiff
#'
#' Returns a new function that computes the gradient of \code{f} with respect
#' to its first argument (or all arguments if \code{argnums} specifies multiple).
#'
#' @param f An R function using supported XLA operations.
#' @param dtype Default dtype: "f32" or "f64".
#' @param argnum Which argument to differentiate with respect to (0-indexed).
#'   Defaults to 0 (first argument).
#' @return A function that computes gradients.
#' @export
xla_grad <- function(f, dtype = "f32", argnum = 0L) {
  cache <- new.env(parent = emptyenv())
  client <- NULL

  function(...) {
    if (is.null(client)) client <<- xla_client()
    args <- list(...)

    key <- paste(vapply(args, function(a) {
      paste(length(a), dtype, sep = ":")
    }, character(1)), collapse = ",")

    if (is.null(cache[[key]])) {
      cache[[key]] <- build_grad_executable(f, args, dtype, argnum, client)
    }

    bufs <- lapply(args, function(a) {
      d <- as.integer(if (length(a) == 1) integer(0) else length(a))
      xla_buffer(client, as.numeric(a), d, dtype)
    })
    out <- xla_execute(cache[[key]], bufs)
    as.array(out[[1]])
  }
}

# Build the gradient computation and compile it
build_grad_executable <- function(f, args, dtype, argnum, client) {
  builder <- xla_builder("grad")

  # Create parameters
  params <- lapply(seq_along(args), function(i) {
    d <- as.integer(if (length(args[[i]]) == 1) integer(0) else length(args[[i]]))
    p <- rjax_parameter(builder, i - 1L, dtype, d)
    tag_op(p, builder)
  })

  # Forward pass with tape recording
  tape_start()
  # Give parameters tape IDs so gradients can flow to them
  for (i in seq_along(params)) {
    params[[i]] <- tape_record("param", params[[i]], list())
  }
  result <- do.call(f, params)
  nodes <- tape_stop()

  # Seed: gradient of output = 1 (scalar) or ones (tensor)
  result_shape <- op_shape(result)
  if (length(result_shape$dims) == 0) {
    seed <- tag_op(rjax_constant_scalar(builder, 1.0, dtype), builder)
  } else {
    seed <- tag_op(
      rjax_constant_broadcast(builder, 1.0, dtype, as.integer(result_shape$dims)),
      builder
    )
  }

  # Backward pass: accumulate gradients
  grads <- new.env(parent = emptyenv())
  result_id <- tape_id(result)
  if (!is.null(result_id)) {
    grads[[as.character(result_id)]] <- seed
  }

  for (node in rev(nodes)) {
    node_id <- as.character(node$id)
    g <- grads[[node_id]]
    if (is.null(g)) next

    rule <- .vjp_rules[[node$op]]
    if (is.null(rule)) {
      stop("No VJP rule for operation: ", node$op)
    }

    input_grads <- rule(g, node$inputs, node$output)

    for (i in seq_along(node$inputs)) {
      inp <- node$inputs[[i]]
      inp_id <- tape_id(inp)
      if (is.null(inp_id)) next
      inp_key <- as.character(inp_id)
      if (is.null(grads[[inp_key]])) {
        grads[[inp_key]] <- input_grads[[i]]
      } else {
        grads[[inp_key]] <- grads[[inp_key]] + input_grads[[i]]
      }
    }
  }

  # Get gradient for the target parameter
  target_param <- params[[argnum + 1L]]
  target_id <- tape_id(target_param)
  if (is.null(target_id)) {
    stop("Target parameter has no tape ID (not used in computation?)")
  }
  grad_op <- grads[[as.character(target_id)]]
  if (is.null(grad_op)) {
    stop("No gradient flows to parameter ", argnum)
  }

  comp <- xla_build(builder, grad_op)
  xla_compile(client, comp)
}
