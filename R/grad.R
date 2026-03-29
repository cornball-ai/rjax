#' Create a gradient function via reverse-mode autodiff
#'
#' Returns a new function that computes the gradient of \code{f} with respect
#' to its first argument. The returned function is itself traceable, so
#' \code{xla_grad(xla_grad(f))} computes second derivatives.
#'
#' @param f An R function using supported XLA operations.
#' @param dtype Default dtype: "f32" or "f64".
#' @param argnum Which argument to differentiate with respect to (0-indexed).
#' @return A function that computes gradients.
#' @export
xla_grad <- function(f, dtype = "f32", argnum = 0L, backend = "cpu") {
  cache <- new.env(parent = emptyenv())
  client <- NULL

  grad_fn <- function(...) {
    args <- list(...)

    # Trace mode: if args are xla_ops, build gradient ops on the same builder
    if (length(args) > 0 && inherits(args[[1]], "xla_op")) {
      return(build_grad_ops(f, args, dtype, argnum))
    }

    # Execute mode: build, compile, cache, execute
    if (is.null(client)) client <<- xla_client(backend)

    key <- paste(vapply(args, function(a) {
      paste(length(a), dtype, sep = ":")
    }, character(1)), collapse = ",")

    if (is.null(cache[[key]])) {
      cache[[key]] <- build_grad_executable(f, args, dtype, argnum, client)
    }

    bufs <- lapply(args, function(a) {
      d <- infer_dims(a)
      xla_buffer(client, as.numeric(a), d, dtype)
    })
    out <- xla_execute(cache[[key]], bufs)
    as.array(out[[1]])
  }

  grad_fn
}

# Infer XLA dims from an R object
infer_dims <- function(x) {
  d <- dim(x)
  if (!is.null(d)) return(as.integer(d))
  if (length(x) == 1) return(integer(0))
  as.integer(length(x))
}

# Build gradient ops on an existing builder (trace mode).
# Takes xla_op parameters, returns the gradient xla_op.
build_grad_ops <- function(f, params, dtype, argnum) {
  # Save the current tape_ids of params (from the outer tape level).
  # The forward pass (f) may overwrite these (if f is itself a grad fn).
  # After the forward pass, we remap any references to the overwritten
  # IDs back to our param IDs.
  param_ids <- integer(length(params))
  old_ids <- integer(length(params))
  for (i in seq_along(params)) {
    old_ids[i] <- as.integer(attr(params[[i]], "tape_id") %||% 0L)
    id <- next_tape_id()
    attr(params[[i]], "tape_id") <- id
    param_ids[i] <- id
  }

  # Forward pass: record operations on a fresh tape
  tape_start()
  result <- do.call(f, params)
  nodes <- tape_stop()

  # After forward, params may have been re-ID'd by nested grads.
  # Restore our param IDs and build a remap table.
  remap <- new.env(parent = emptyenv())
  for (i in seq_along(params)) {
    cur_id <- as.integer(attr(params[[i]], "tape_id") %||% 0L)
    if (cur_id != param_ids[i] && cur_id != 0L) {
      # The nested grad gave this param a new ID. Map new -> ours.
      remap[[as.character(cur_id)]] <- param_ids[i]
    }
    attr(params[[i]], "tape_id") <- param_ids[i]
  }

  b <- attr(params[[1]], "builder")

  # Seed gradient
  result_shape <- op_shape(result)
  if (length(result_shape$dims) == 0) {
    seed <- tag_op(rjax_constant_scalar(b, 1.0, dtype), b)
  } else {
    seed <- tag_op(
      rjax_constant_broadcast(b, 1.0, dtype, as.integer(result_shape$dims)),
      b
    )
  }

  # Backward pass: walk tape in reverse, accumulate gradients
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
    if (is.null(rule)) stop("No VJP rule for operation: ", node$op)

    input_grads <- rule(g, node$inputs, node$output)

    for (i in seq_along(node$inputs)) {
      inp_id <- node$input_ids[i]
      if (is.na(inp_id)) next
      # Remap IDs from nested grads back to our param IDs
      remapped <- remap[[as.character(inp_id)]]
      if (!is.null(remapped)) inp_id <- remapped
      inp_key <- as.character(inp_id)
      if (is.null(grads[[inp_key]])) {
        grads[[inp_key]] <- input_grads[[i]]
      } else {
        grads[[inp_key]] <- grads[[inp_key]] + input_grads[[i]]
      }
    }
  }

  target_id <- param_ids[argnum + 1L]
  grad_op <- grads[[as.character(target_id)]]
  if (is.null(grad_op)) stop("No gradient flows to parameter ", argnum)

  grad_op
}

# Build and compile the gradient computation (execute mode)
build_grad_executable <- function(f, args, dtype, argnum, client) {
  builder <- xla_builder("grad")

  params <- lapply(seq_along(args), function(i) {
    d <- infer_dims(args[[i]])
    p <- rjax_parameter(builder, i - 1L, dtype, d)
    tag_op(p, builder)
  })

  grad_op <- build_grad_ops(f, params, dtype, argnum)

  comp <- xla_build(builder, grad_op)
  xla_compile(client, comp)
}
