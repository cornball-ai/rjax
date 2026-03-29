# Tape stack for recording operations during tracing.
# Supports nesting for grad-of-grad.

.tape <- new.env(parent = emptyenv())
.tape$stack <- list()
.tape$global_counter <- 0L

tape_start <- function() {
  frame <- list(nodes = list())
  .tape$stack <- c(.tape$stack, list(frame))
}

tape_stop <- function() {
  n <- length(.tape$stack)
  if (n == 0) stop("tape_stop called with no active tape")
  frame <- .tape$stack[[n]]
  .tape$stack <- .tape$stack[-n]
  frame$nodes
}

tape_is_recording <- function() {
  length(.tape$stack) > 0
}

next_tape_id <- function() {
  .tape$global_counter <- .tape$global_counter + 1L
  .tape$global_counter
}

# Record an operation on the top-most tape.
# Snapshots input tape_ids so they're stable even if nested grads
# overwrite the live attribute on shared external pointers.
tape_record <- function(op_name, output, inputs, extras = list()) {
  n <- length(.tape$stack)
  if (n == 0) return(output)

  id <- next_tape_id()
  attr(output, "tape_id") <- id

  # Snapshot input IDs at record time (immune to later mutation)
  input_ids <- vapply(inputs, function(inp) {
    tid <- attr(inp, "tape_id")
    if (is.null(tid)) NA_integer_ else as.integer(tid)
  }, integer(1))

  frame <- .tape$stack[[n]]
  frame$nodes[[length(frame$nodes) + 1L]] <- list(
    id = id,
    op = op_name,
    output = output,
    inputs = inputs,
    input_ids = input_ids,
    extras = extras
  )
  .tape$stack[[n]] <- frame
  output
}

tape_id <- function(op) {
  attr(op, "tape_id")
}
