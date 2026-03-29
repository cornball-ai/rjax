# Tape for recording operations during tracing (used by autodiff)

.tape <- new.env(parent = emptyenv())
.tape$recording <- FALSE
.tape$nodes <- NULL
.tape$counter <- 0L

tape_start <- function() {
  .tape$recording <- TRUE
  .tape$nodes <- list()
  .tape$counter <- 0L
}

tape_stop <- function() {
  .tape$recording <- FALSE
  nodes <- .tape$nodes
  .tape$nodes <- NULL
  nodes
}

tape_is_recording <- function() {
  .tape$recording
}

# Record an operation on the tape.
# Returns the output op, tagged with a tape ID.
tape_record <- function(op_name, output, inputs, extras = list()) {
  if (!.tape$recording) return(output)
  .tape$counter <- .tape$counter + 1L
  id <- .tape$counter
  attr(output, "tape_id") <- id
  .tape$nodes[[length(.tape$nodes) + 1L]] <- list(
    id = id,
    op = op_name,
    output = output,
    inputs = inputs,
    extras = extras
  )
  output
}

tape_id <- function(op) {
  attr(op, "tape_id")
}
