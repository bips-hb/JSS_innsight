
###############################################################################
#                           Keras dense models
###############################################################################

get_dense_model <- function(shape, name, save = TRUE, act_name = "relu",
                            bias = TRUE, num_outputs = 5, src_dir = "models",
                            depth = 1, width = 64) {
  library(keras)
  library(tensorflow)
  k_clear_session()

  config <- tf$compat$v1$ConfigProto(intra_op_parallelism_threads = 1L,
                                     inter_op_parallelism_threads = 1L)
  session = tf$compat$v1$Session(config=config)
  tf$compat$v1$keras$backend$set_session(session)


  ## Define model
  # first layer
  model <- keras_model_sequential(input_shape = shape) %>%
    layer_dense(units = width, activation = act_name, use_bias = bias,
                bias_initializer = "glorot_uniform")

  for (i in seq_len(depth - 1)) {
    model %>%
      layer_dense(units = width, activation = act_name, use_bias = bias,
                  bias_initializer = "glorot_uniform")
  }

  # last layer
  model %>%
    layer_dense(units = num_outputs, activation = "linear", use_bias = bias,
                bias_initializer = "glorot_uniform")

  # compile model
  model %>% compile("adam", loss = loss_mean_squared_error())

  if (save) {
    save_model_hdf5(model, paste0(src_dir, "/models/", name))
  }

  k_clear_session()

  model
}

###############################################################################
#                           Keras CNN models
###############################################################################

get_2D_model <- function(shape, name, save = TRUE, act_name = "relu",
                         bias = TRUE, pooling = "none", bn = "none",
                         num_outputs = 5, src_dir = "models") {
  library(keras)
  library(tensorflow)
  k_clear_session()

  config <- tf$compat$v1$ConfigProto(intra_op_parallelism_threads = 1L,
                           inter_op_parallelism_threads = 1L)
  session = tf$compat$v1$Session(config=config)
  tf$compat$v1$keras$backend$set_session(session)

  # Define model
  model <- keras_model_sequential(input_shape = shape) %>%
    layer_conv_2d(filters = 5,
                  strides = if (pooling == "none") c(2L, 2L) else c(1L, 1L),
                  kernel_size = c(4,4),
                  activation = if (bn == "before_act") "linear" else act_name,
                  bias_initializer = "glorot_uniform", use_bias = bias)

  # Add batch normalization
  if (bn == "after_act") {
    model %>%
      layer_batch_normalization(
        #center = bias,
        beta_initializer = if (bias) init() else "zeros",
        gamma_initializer = init(),
        moving_mean_initializer = if (bias) init() else "zeros",
        moving_variance_initializer = if (bias) init_plus() else "ones"
      )
  } else if (bn == "before_act") {
    model %>%
      layer_batch_normalization(
        #center = bias,
        beta_initializer = if (bias) init() else "zeros",
        gamma_initializer = init(),
        moving_mean_initializer = if (bias) init() else "zeros",
        moving_variance_initializer = if (bias) init_plus() else "ones"
      ) %>%
      layer_activation(activation = act_name)
  }

  # add pooling
  if (pooling == "avg") {
    model %>% layer_average_pooling_2d(pool_size = c(3L, 3L))
  } else if (pooling == "max") {
    model %>% layer_max_pooling_2d(pool_size = c(3L, 3L))
  }

  # flatten and add dense layer
  model %>%
    layer_flatten() %>%
    layer_dense(units = num_outputs, activation = "linear", use_bias = bias,
                bias_initializer = "glorot_uniform")

  # compile model
  model %>% compile("adam", loss = loss_mean_squared_error())

  if (save) {
    save_model_hdf5(model, paste0(src_dir,"/models/", name))
  }

  k_clear_session()

  model
}


###############################################################################
#                           Utility functions
###############################################################################
init <- function() {
  initializer_random_uniform(-0.5, 0.5)
}

init_plus <- function() {
  initializer_random_uniform(0.8, 1.2)
}
