
###############################################################################
#        Preprocessing:
#                 - Create models from 'config'
#                 - Create inputs
###############################################################################

preprocess <- function(num_models, config, src_dir = "tmp") {
  library(callr)
  library(cli)

  cli_h1("Pre-processing")

  # delete old models, inputs and config file
  unlink(paste0(src_dir, c("/figures*", "/inputs*", "/model_config*", "/models*",
                           "/results*")), recursive = TRUE)

  # create directory
  dir.create(src_dir, showWarnings = FALSE)

  # ------- Create model config -----------------------------------------------
  cli_progress_step("Creating model configuration file")
  # tabular config
  config_tab <- config$config_tab
  if (nrow(config_tab) > 0) {
    config_tab$batchnorm <- "none"
    config_tab$pooling <- "none"
    config_tab$data_type <- "tabular"
    config_tab$model_name <- paste0("dense_", seq_len(nrow(config_tab)))
  }

  # image config
  config_2D <- config$config_2D
  if (nrow(config_2D) > 0) {
    config_2D$data_type <- "image"
    config_2D$model_name <- paste0("2DCNN_", seq_len(nrow(config_2D)))
  }

  # create model config
  model_config <- rbind(config_tab, config_2D)
  model_config <-
    model_config[rep(seq_len(nrow(model_config)), each = num_models), ]
  model_config$model_name <-
    paste0(model_config$model_name, "_", seq_len(num_models))
  model_config <- rbind(model_config, model_config)
  model_config$api <- rep(c("torch", "keras"), each = nrow(model_config) %/% 2)
  model_config$model_name <- paste0(
    model_config$model_name,
    rep(c("_torch.pt", "_keras.h5"), each = nrow(model_config) %/% 2)
  )
  row.names(model_config) <- NULL

  # save model config
  saveRDS(model_config, paste0(src_dir, "/model_config.rds"))

  # ------- Create models -----------------------------------------------------
  cli_progress_step("Creating {.pkg torch} models")
  dir.create(paste0(src_dir, "/models"))
  # create PyTorch models
  r(create_torch_models,
    args = list(model_config[model_config$api == "torch", ], src_dir))
  # create keras models
  cli_progress_step("Creating {.pkg keras} models")
  r(create_keras_models,
    args = list(model_config[model_config$api == "keras", ], src_dir))

  # ------- Create inputs -----------------------------------------------------
  cli_progress_step("Creating inputs")
  generate_inputs(model_config, src_dir)


  model_config
}

###############################################################################
#                         Create PyTorch models
###############################################################################

create_torch_models <- function(config, src_dir) {
  # load conda env and PyTorch
  reticulate::use_condaenv("JSS_innsight_pytorch")
  py_torch <- reticulate::import("torch")

  # set seeds
  set.seed(42)
  py_torch$manual_seed(42)

  # create models
  py_method = reticulate::import_from_path("models_pytorch", here::here("utils/preprocess/"))
  for (i in seq_len(nrow(config))) {
    config_i <- config[i, ]
    if (config_i$data_type == "tabular") {
      py_method$get_dense_model(
        config_i$input_shape[[1]][-1], config_i$model_name, save = TRUE,
        act = config_i$act, bias = config_i$bias,
        num_outputs = config_i$num_outputs, src_dir = src_dir)
    } else if (config_i$data_type == "image") {
      py_method$get_2D_model(
        config_i$input_shape[[1]][-1][c(3,1,2)], config_i$model_name,
        save = TRUE, act = config_i$act, bias = config_i$bias,
        pooling = config_i$pooling, bn = config_i$batchnorm,
        num_outputs = config_i$num_outputs, src_dir = src_dir)
    }
  }
}

###############################################################################
#                         Create Keras models
###############################################################################

create_keras_models <- function(config, src_dir) {
  # Set up seeds and python environment
  set.seed(42)
  Sys.setenv("RETICULATE_PYTHON" = reticulate::conda_python("JSS_innsight_tf_1"))
  reticulate::use_condaenv("JSS_innsight_tf_1")
  keras::use_condaenv("JSS_innsight_tf_1")

  library(keras)
  library(tensorflow)
  tensorflow::tf$set_random_seed(123)

  # craete models
  source("utils/preprocess/models_keras.R")
  for (i in seq_len(nrow(config))) {
    config_i <- config[i, ]
    if (config_i$data_type == "tabular") {
      get_dense_model(
        config_i$input_shape[[1]][-1], config_i$model_name, save = TRUE,
        act = config_i$act, bias = config_i$bias,
        num_outputs = config_i$num_outputs, src_dir = src_dir)
    } else if (config_i$data_type == "image") {
      get_2D_model(
        config_i$input_shape[[1]][-1], config_i$model_name,
        save = TRUE, act = config_i$act, bias = config_i$bias,
        pooling = config_i$pooling, bn = config_i$batchnorm,
        num_outputs = config_i$num_outputs, src_dir = src_dir)
    }
  }
}

###############################################################################
#                         Generate inputs
###############################################################################

generate_inputs <- function(config, src_dir) {
  set.seed(42)
  dir.create(paste0(src_dir, "/inputs"))

  for (shape in unique(config$input_shape)) {
    # channels last
    inputs <- array(rnorm(prod(shape)), dim = shape)
    saveRDS(inputs, paste0(src_dir, "/inputs/input_", paste(shape, collapse = "_"), "_last.rds"))
    inputs_ref <- array(rnorm(prod(shape)), dim = c(1, shape[-1]))
    saveRDS(inputs_ref, paste0(src_dir, "/inputs/input_ref_", paste(shape, collapse = "_"), "_last.rds"))

    # channels first
    shape <- append(shape[-length(shape)], shape[length(shape)], after = 1)
    inputs <- array(rnorm(prod(shape)), dim = shape)
    saveRDS(inputs, paste0(src_dir, "/inputs/input_", paste(shape, collapse = "_"), "_first.rds"))
    inputs_ref <- array(rnorm(prod(shape)), dim = c(1, shape[-1]))
    saveRDS(inputs_ref, paste0(src_dir, "/inputs/input_ref_", paste(shape, collapse = "_"), "_first.rds"))
  }
}
