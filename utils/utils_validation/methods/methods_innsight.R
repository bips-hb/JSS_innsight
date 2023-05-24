
###############################################################################
#                       innsight: Gradient
###############################################################################
apply_Gradient <- function(model, inputs, func_args = NULL, num_outputs = NULL, n_cpu = NULL) {
  library(torch)

  if (torch::is_nn_module(model)) {
    input_dim <- dim(inputs)[-1]
  } else {
    input_dim <- NULL
  }
  start_time <- Sys.time()
  c <- Converter$new(model, input_dim = input_dim)
  convert_time <- as.numeric(Sys.time() - start_time)

  input_time <- Sys.time()
  grad <- Gradient$new(c, inputs, times_input = func_args$times_input,
                       channels_first = func_args$channels_first,
                       output_idx = seq_len(num_outputs))$result[[1]][[1]]
  end_time <- Sys.time()

  list(
    total_time = as.numeric(end_time - start_time),
    eval_time = as.numeric(end_time - input_time),
    convert_time = convert_time,
    result =  torch::as_array(grad))
}

###############################################################################
#                       innsight: SmoothGrad
###############################################################################
apply_SmoothGrad <- function(model, inputs, func_args = NULL, num_outputs = NULL, n_cpu = NULL) {
  library(torch)

  if (torch::is_nn_module(model)) {
    input_dim <- dim(inputs)[-1]
  } else {
    input_dim <- NULL
  }
  start_time <- Sys.time()
  c <- Converter$new(model, input_dim = input_dim)
  convert_time <- as.numeric(Sys.time() - start_time)

  input_time <- Sys.time()
  grad <- SmoothGrad$new(c, inputs, times_input = func_args$times_input,
                       channels_first = func_args$channels_first,
                       n = func_args$n, noise_level = func_args$noise_level,
                       output_idx = seq_len(num_outputs))$result[[1]][[1]]
  end_time <- Sys.time()

  list(
    total_time = as.numeric(end_time - start_time),
    eval_time = as.numeric(end_time - input_time),
    convert_time = convert_time,
    result =  torch::as_array(grad))
}

###############################################################################
#                       innsight: LRP
###############################################################################
apply_LRP <- function(model, inputs, func_args = NULL, num_outputs = NULL, n_cpu = NULL) {
  library(torch)

  if (torch::is_nn_module(model)) {
    input_dim <- dim(inputs)[-1]
  } else {
    input_dim <- NULL
  }

  start_time <- Sys.time()
  c <- Converter$new(model, input_dim = input_dim)
  convert_time <- as.numeric(Sys.time() - start_time)

  input_time <- Sys.time()
  grad <- LRP$new(c, inputs, rule_name = func_args$rule_name,
                  rule_param = func_args$rule_param,
                  channels_first = func_args$channels_first,
                  output_idx = seq_len(num_outputs))$result[[1]][[1]]
  end_time <- Sys.time()

  list(
    total_time = as.numeric(end_time - start_time),
    eval_time = as.numeric(end_time - input_time),
    convert_time = convert_time,
    result = torch::as_array(grad))
}

###############################################################################
#                       innsight: DeepLift
###############################################################################

apply_DeepLift <- function(model, inputs, func_args = NULL, num_outputs = NULL, n_cpu = NULL) {
  library(torch)

  if (torch::is_nn_module(model)) {
    input_dim <- dim(inputs)[-1]
  } else {
    input_dim <- NULL
  }
  start_time <- Sys.time()
  c <- Converter$new(model, input_dim = input_dim)
  convert_time <- as.numeric(Sys.time() - start_time)

  input_time <- Sys.time()
  grad <- DeepLift$new(c, inputs, rule_name = func_args$rule_name, x_ref = func_args$x_ref,
                       channels_first = func_args$channels_first,
                       output_idx = seq_len(num_outputs))$result[[1]][[1]]
  end_time <- Sys.time()

  list(
    total_time = as.numeric(end_time - start_time),
    eval_time = as.numeric(end_time - input_time),
    convert_time = convert_time,
    result = torch::as_array(grad))
}


