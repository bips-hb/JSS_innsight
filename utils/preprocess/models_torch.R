
###############################################################################
#                           Torch dense models
###############################################################################

get_dense_model <- function(shape, act, bias, num_outputs) {
  library(torch)

  ## Define model
  # first layer
  model <- nn_sequential(nn_linear(shape, 64, bias = bias))
  if (act == "relu") {
    model$add_module("first_act", nn_relu())
  } else if (act == "tanh") {
    model$add_module("first_act", nn_tanh())
  }

  # second layer
  model$add_module("second_layer", nn_linear(64, num_outputs, bias = bias))

  model
}

###############################################################################
#                           Torch CNN models
###############################################################################

get_2D_model <- function(shape, act = "relu", bias = TRUE, pooling = "none",
                         bn = "none", num_outputs = 5) {
  library(torch)

  in_channels <- shape[1]

  # Define activation function
  if (act == "relu") {
    activation <- nn_relu
  } else if (act == "tanh") {
    activation <- nn_tanh
  }

  ## Define model
  model <- nn_sequential(
    nn_conv2d(in_channels, 5, c(4,4), bias = bias)
  )
  if (bn == "none") {
    model$add_module("act_1", activation())
  } else if (bn == "after_act") {
    model$add_module("act_1", activation())
    model$add_module("batchnorm_1", nn_batch_norm2d(5, affine = bias))
  } else if (bn == "before_act") {
    model$add_module("batchnorm_1", nn_batch_norm2d(5, affine = bias))
    model$add_module("act_1", activation())
  }

  if (pooling == "avg") {
    model$add_module("AvgPooling_1", nn_avg_pool2d(c(3,3)))
  } else if (pooling == "max") {
    model$add_module("MaxPooling_1", nn_max_pool2d(c(3,3)))
  }
  model$add_module("Flatten", nn_flatten())

  # Calc output of flatten
  inputs <- torch_randn(c(1, shape))
  out <- model(inputs)

  # Add last layer
  model$add_module("output", nn_linear(out$shape[2], num_outputs, bias = bias))

  model$eval()

  model
}
