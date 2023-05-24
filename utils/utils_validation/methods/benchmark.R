
PKG_METHODS <- list(
  zennit = c("Gradient", "GradxInput", "SmoothGrad", "LRP (simple)",
             "LRP (epsilon)", "LRP (alpha-beta)"),
  captum = c("Gradient", "GradxInput", "SmoothGrad", "LRP (simple)",
             "LRP (epsilon)", "DeepLift (rescale)"),
  innvestigate = c("Gradient", "GradxInput", "LRP (simple)",
                   "LRP (epsilon)", "LRP (alpha-beta)"),
  deeplift = c("Gradient", "GradxInput", "DeepLift (rescale)",
               "DeepLift (reveal-cancel)"),
  innsight_keras = c("Gradient", "GradxInput", "SmoothGrad", "LRP (simple)",
                     "LRP (epsilon)", "LRP (alpha-beta)", "DeepLift (rescale)",
                     "DeepLift (reveal-cancel)"),
  innsight_torch = c("Gradient", "GradxInput", "SmoothGrad", "LRP (simple)",
                     "LRP (epsilon)", "LRP (alpha-beta)", "DeepLift (rescale)",
                     "DeepLift (reveal-cancel)")
)

LRP_RULE_SPEC <- list(
  zennit = data.frame(
    layer = c("BatchNorm_Layer", "AvgPool2D_Layer"),
    rule_name = c("epsilon", "epsilon"),
    rule_param = c(1e-10, 1e-10)),
  captum = data.frame(
    layer = c("BatchNorm_Layer"),
    rule_name = c("epsilon"),
    rule_param = c(1e-10)),
  innvestigate = data.frame(
    layer = c("AvgPool2D_Layer"),
    rule_name = c("simple"),
    rule_param = c(0))
)

benchmark <- function(methods, model_config, show = FALSE, src_dir = "tmp", n_cpu = 1) {
  library(callr)
  library(cli)

  if (!dir.exists(paste0(src_dir, "/results"))) {
    dir.create(paste0(src_dir, "/results"))
  }

  cli_h1("Bechmarking")

  # Benchmark innsight vs zennit ----------------------------------------------
  cli_h3("Benchmarking {.pkg innsight} vs. {.pkg zennit}")
  cli_progress_step("Calculating {.pkg innsight}")
  res_innsight <- r(
    apply_benchmark,
    args = list(pkg = "innsight_torch", methods = methods, ref_pkg = "zennit",
                model_config = model_config[model_config$api == "torch", ],
                rule_spec = LRP_RULE_SPEC$zennit, src_dir = src_dir, n_cpu = n_cpu),
    show = show, spinner = FALSE)
  cli_progress_step("Calculating {.pkg zennit}")
  res_zennit <- r(
    apply_benchmark,
    args = list(pkg = "zennit", methods = methods, ref_pkg = "innsight_torch",
                model_config = model_config[model_config$api == "torch", ],
                src_dir = src_dir, n_cpu = n_cpu),
    show = show, spinner = FALSE)

  # Benchmark innsight vs captum ----------------------------------------------
  cli_h3("Benchmarking {.pkg innsight} vs. {.pkg captum}")
  cli_progress_step("Calculating {.pkg innsight}")
  res_innsight <- r(
    apply_benchmark,
    args = list(pkg = "innsight_torch", methods = methods, ref_pkg = "captum",
                model_config = model_config[model_config$api == "torch", ],
                rule_spec = LRP_RULE_SPEC$captum, src_dir = src_dir, n_cpu = n_cpu),
    show = show, spinner = FALSE)
  cli_progress_step("Calculating {.pkg captum}")
  res_captum <- r(
    apply_benchmark,
    args = list(pkg = "captum", methods = methods, ref_pkg = "innsight_torch",
                model_config = model_config[model_config$api == "torch", ],
                src_dir = src_dir, n_cpu = n_cpu),
    show = show, spinner = FALSE)

  # Benchmark innsight vs innvestigate -----------------------------------------
  cli_h3("Benchmarking {.pkg innsight} vs. {.pkg innvestigate}")
  cli_progress_step("Calculating {.pkg innsight}")
  res_innsight <- r(
    apply_benchmark,
    args = list(pkg = "innsight_keras", methods = methods, ref_pkg = "innvestigate",
                model_config = model_config[model_config$api == "keras", ],
                rule_spec = LRP_RULE_SPEC$innvestigate,
                src_dir = src_dir, n_cpu = n_cpu),
    show = show, spinner = FALSE)
  cli_progress_step("Calculating {.pkg innvestigate}")
  res_innvestigate <- r(
    apply_benchmark,
    args = list(pkg = "innvestigate", methods = methods, ref_pkg = "innsight_keras",
                model_config = model_config[model_config$api == "keras", ],
                src_dir = src_dir, n_cpu = n_cpu),
    show = show, spinner = FALSE)

  # Benchmark innsight vs deeplift -----------------------------------------
  cli_h3("Benchmarking {.pkg innsight} vs. {.pkg deeplift}")
  cli_progress_step("Calculating {.pkg innsight}")
  res_innsight <- r(
    apply_benchmark,
    args = list(pkg = "innsight_keras", methods = methods, ref_pkg = "deeplift",
                model_config = model_config[model_config$api == "keras", ],
                rule_spec = LRP_RULE_SPEC$deeplift, src_dir = src_dir, n_cpu = n_cpu),
    show = show, spinner = FALSE)
  cli_progress_step("Calculating {.pkg deeplift}")
  res_deeplift <- r(
    apply_benchmark,
    args = list(pkg = "deeplift", methods = methods, ref_pkg = "innsight_keras",
                model_config = model_config[model_config$api == "keras", ],
                src_dir = src_dir, n_cpu = n_cpu),
    show = show, spinner = FALSE)
  cli_progress_cleanup()

  result <- get_results(src_dir)


  # Set the order
  result[[1]]$method_grp[result[[1]]$method_grp == "Gradient"] <- "Gradient-based"
  result[[2]]$method_grp[result[[2]]$method_grp == "Gradient"] <- "Gradient-based"
  result[[1]]$bias <- ifelse(result[[1]]$bias, "with bias", "no bias")
  result[[2]]$bias <- ifelse(result[[2]]$bias, "with bias", "no bias")
  result[[1]]$method <- factor(result[[1]]$method,
                               levels = unique(result[[1]]$method))
  result[[2]]$method <- factor(result[[2]]$method,
                               levels = unique(result[[2]]$method))

  # Save result
  saveRDS(result, paste0(src_dir, "/results.rds"))

  result
}

apply_benchmark <- function(pkg, methods, model_config, ref_pkg,
                            rule_spec = NULL, src_dir = "models", n_cpu = 1) {
  library(data.table)
  library(cli)
  set.seed(42)
  cli_text("")

  # Disable GPU usage
  Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)

  results <- data.table()
  source("utils/utils_validation/methods/benchmark.R")

  # Load conda environments
  load_conda_envs(pkg, n_cpu)

  for (method in methods) {
    # deparse method argument
    d_method <- deparse_method(method)

    if (d_method$method %in% PKG_METHODS[[pkg]] & d_method$method %in% PKG_METHODS[[ref_pkg]]) {
      start_time <- Sys.time()

      result <- data.table(model_config)
      result$method_grp <- d_method$m_group
      result$method <- d_method$method
      result$method_arg <- d_method$m_arg
      result$pkg <- strsplit(pkg, "_")[[1]][1]
      result$time_total <- NA_real_
      result$time_eval <- NA_real_
      result$time_convert <- NA_real_
      result$result <- list(list(NA_real_))

      # Load method function for selected package -----------------------------
      if (startsWith(pkg, "innsight")) {
        source("utils/utils_validation/methods/methods_innsight.R")
        func <- switch(d_method$m_group,
                       Gradient = apply_Gradient,
                       SmoothGrad = apply_SmoothGrad,
                       LRP = apply_LRP,
                       DeepLift = apply_DeepLift)

        # Add rule specifications
        if (d_method$m_group == "LRP") {
          rule <- get_rule(d_method$func_args$rule_name,
                           d_method$func_args$rule_param,
                           rule_spec)
          d_method$func_args$rule_name <- rule[[1]]
          d_method$func_args$rule_param <- rule[[2]]
        }
      } else {
        py_methods <- reticulate::import_from_path(paste0("methods_", pkg),
                                                   here::here("utils/utils_validation/methods/"))
        func <- py_methods[[paste0("apply_", d_method$m_group)]]
      }

      for (i in seq_len(nrow(model_config))) {
        config_i <- model_config[i, ]
        func_args <- d_method$func_args
        # Load model/model_path ------------------------------------------------
        if (pkg %in% c("innvestigate", "deeplift")) {
          model <- paste0(src_dir, "/models/", config_i$model_name)
        } else if (pkg %in% c("zennit", "captum")) {
          model <- load_pytorch_model(config_i, src_dir)
        } else if (pkg == "innsight_torch") {
          res <- load_torch_model(config_i, src_dir)
          model <- res$model
          func_args$input_dim <- res$input_dim
          func_args$channels_first <- TRUE
        } else if (pkg == "innsight_keras") {
          reticulate::py_capture_output({
            library(keras)
            model <- load_model_hdf5(paste0(src_dir, "/models/", config_i$model_name))
          })
          func_args$channels_first <- FALSE
        }

        # Load inputs ----------------------------------------------------------
        if (config_i$api == "keras") {
          shape <- paste(config_i$input_shape[[1]], collapse = "_")
          inputs <- readRDS(paste0(src_dir, "/inputs/input_", shape, "_last.rds"))
          inputs_ref <- readRDS(paste0(src_dir, "/inputs/input_ref_", shape, "_last.rds"))
        } else if (config_i$api == "torch") {
          shape <- config_i$input_shape[[1]]
          shape <- append(shape[-length(shape)], shape[length(shape)], after = 1)
          shape <- paste(shape, collapse = "_")
          inputs <- readRDS(paste0(src_dir, "/inputs/input_", shape, "_first.rds"))
          inputs_ref <- readRDS(paste0(src_dir, "/inputs/input_ref_", shape, "_first.rds"))
        }

        # add reference value for DeepLift
        if (startsWith(method, "DeepLift")) {
          if (func_args$x_ref == "zeros") {
            func_args$x_ref <- inputs_ref * 0
          } else if (func_args$x_ref == "norm") {
            func_args$x_ref <- inputs_ref
          }
        }

        # Apply method --------------------------------------------------------
        tryCatch(
          expr = {
            reticulate::py_capture_output(
              res <- func(model, inputs, func_args, config_i$num_outputs, n_cpu),
              type = "stdout")
          },
          error = function(e){
            warning("Look at index: ", i)
            print(e)
            res <- list(total_time = NA, eval_time = NA,
                        convert_time = NA, result = array(NA))
            res
          })
        # Add results in data.table
        result[i, "time_total"] <- res$total_time
        result[i, "time_eval"] <- res$eval_time
        result[i, "time_convert"] <- res$convert_time
        result[i, "result"] <- list(list(list(as.array(res$result))))
      }

      time_diff <- Sys.time() - start_time
      time_str <- col_grey(paste0(" [", round(time_diff, 1), attributes(time_diff)$units, "]"))
      cli_bullets(c("v" = paste0(d_method$method, time_str)))

      saveRDS(result,
              file = paste0(src_dir, "/results/result-", method, "-", pkg, "-",
                            ref_pkg, ".rds"))
    }
  }

}


###############################################################################
#                           Utility functions
###############################################################################

deparse_method <- function(method) {
  args <- unlist(strsplit(method, "_"))
  if (startsWith(method, "Grad")) {
    times_input <- if (method == "Gradient") FALSE else TRUE
    res <- list(
      m_group = "Gradient",
      method = method,
      m_arg = "",
      func_args = list(times_input = times_input))
  } else if (startsWith(method, "LRP")) {
    res <- list(
      m_group = "LRP",
      method = paste0(args[[1]], " (", args[[2]],")"),
      m_arg = args[[3]],
      func_args = list(
        rule_name = gsub("-", "_", args[[2]]),
        rule_param = as.numeric(args[[3]])
        )
      )
  } else if (startsWith(method, "DeepLift")) {
    res <- list(
      m_group = "DeepLift",
      method = paste0(args[[1]], " (", args[[2]],")"),
      m_arg = args[[3]],
      func_args = list(rule_name = gsub("-", "_", args[[2]]),
                       x_ref = args[[3]])
    )
  } else if (startsWith(method, "Smooth")) {
    res <- list(
      m_group = "SmoothGrad",
      method = paste0(args[[1]]),
      m_arg = paste0(args[2:3], collapse = "-"),
      func_args = list(
        times_input = FALSE,
        n = as.numeric(args[[2]]),
        noise_level = as.numeric(args[[3]])
      )
    )
  }

  res
}


load_torch_model <- function(config, src_dir) {
  source("utils/utils_validation/preprocess/models_torch.R")
  input_shape <- config$input_shape[[1]][-1]

  # move channels first
  input_shape <- append(input_shape[-length(input_shape)],
                        input_shape[length(input_shape)], after = 0)
  # Get model
  if (config$data_type == "tabular") {
    depth <- config$hidden_depth
    width <- config$hidden_width
    if (is.null(depth)) depth <- 1
    if (is.null(width)) width <- 64

    model <- get_dense_model(input_shape, config$act, config$bias,
                             config$num_outputs, width, depth)
  } else if (config$data_type == "image") {
    depth <- config$hidden_depth
    width <- config$hidden_width
    if (is.null(depth)) depth <- 1
    if (is.null(width)) width <- 5
    model <- get_2D_model(input_shape, config$act, config$bias,
                          config$pooling, config$batchnorm,
                          config$num_outputs, depth, width)
  }

  # Load state dict
  state_dict <- load_state_dict(paste0(src_dir, "/models/", config$model_name))
  model$load_state_dict(state_dict)
  model$eval()

  list(model = model, input_dim = input_shape)
}

load_pytorch_model <- function(config, src_dir) {
  pytorch_models = reticulate::import_from_path("models_pytorch",
                                                here::here("utils/utils_validation/preprocess/"))
  import_torch <- reticulate::import("torch")

  input_shape <- config$input_shape[[1]][-1]

  # move channels first
  input_shape <- append(input_shape[-length(input_shape)],
                        input_shape[length(input_shape)], after = 0)

  # Get model
  if (config$data_type == "tabular") {
    depth <- config$hidden_depth
    width <- config$hidden_width
    if (is.null(depth)) depth <- 1
    if (is.null(width)) width <- 64

    model <- pytorch_models$get_dense_model(input_shape, "", FALSE, config$act,
                                            config$bias, num_outputs = config$num_outputs,
                                            depth = depth, width = width)
  } else if (config$data_type == "image") {
    depth <- config$hidden_depth
    width <- config$hidden_width
    if (is.null(depth)) depth <- 1
    if (is.null(width)) width <- 5
    model <- pytorch_models$get_2D_model(
      input_shape, "", FALSE, config$act, config$bias, config$pooling,
      config$batchnorm, num_outputs = config$num_outputs,
      depth = depth, width = width)
  }

  # Load state dict
  state_dict <- import_torch$load(paste0(src_dir, "/models/", config$model_name))
  model$load_state_dict(state_dict)
  model$eval()

  model
}

get_rule <- function(rule_name, rule_arg, rule_spec) {
  layer_names <- c("Dense_Layer", "Conv2D_Layer", "Conv1D_Layer",
                   "MaxPool1D_Layer", "MaxPool2D_Layer", "AvgPool1D_Layer",
                   "AvgPool2D_Layer", "BatchNorm_Layer")
  rule_name <- rep(list(rule_name), length(layer_names))
  names(rule_name) <- layer_names
  rule_arg <- rep(list(rule_arg), length(layer_names))
  names(rule_arg) <- layer_names

  # Add specifications
  rule_name[rule_spec$layer] <- rule_spec$rule_name
  rule_arg[rule_spec$layer] <- rule_spec$rule_param

  list(rule_name, rule_arg)
}

get_results <- function(src_dir) {
  result_names <- list.files(paste0(src_dir, "/results/"))

  fun <- function(x) {
    s <- strsplit(x, "-")[[1]]
    s[length(s)] <- strsplit(s[length(s)], "[.]")[[1]][1]
    s <- c(s[seq_len(length(s) - 2)], s[length(s)], s[length(s) - 1])
    s <- paste0(paste0(s, collapse = "-"), ".rds")

    combine_results(
      readRDS(paste0(src_dir, "/results/", s)),
      readRDS(paste0(src_dir, "/results/", x))
    )
  }

  library(data.table)

  res_error <- data.table()
  res_time <- data.table()

  pkgs <- c("zennit", "captum", "innvestigate", "deeplift")
  for (pkg in pkgs) {
    res <-
      lapply(
        result_names[grepl(paste0(pkg, "-innsight"), result_names)],
        FUN = fun)

    res_error <- rbind(
      res_error,
      do.call("rbind", lapply(res, function(x) x[[1]]))
    )

    res_time <- rbind(
      res_time,
      do.call("rbind", lapply(res, function(x) x[[2]]))
    )
  }

  list(res_error, res_time)
}

get_results_time <- function(src_dir) {
  library(data.table)
  result_names <- list.files(paste0(src_dir, "/results/"))

  args <- lapply(result_names,
                 function(x) readRDS(paste0(src_dir, "/results/", x))[, -"result"])

  res <- do.call("rbind", args)
  res$method_grp[res$method_grp == "Gradient"] <- "Gradient-based"

  res
}

combine_results <- function(df1, df2) {

  res <- df2[, c("input_shape", "bias", "act", "num_outputs", "batchnorm",
                 "pooling", "data_type", "model_name", "api", "method_grp",
                 "method", "method_arg", "pkg")]

  # Result for time comparison
  res_time <- res
  res_time$time_total <- df1$time_total - df2$time_total
  res_time$time_eval <- df1$time_eval - df2$time_eval
  res_time$time_convert <- df1$time_convert - df2$time_convert

  # Result
  res_comp <- res

  error <- lapply(seq_len(nrow(df1)),
         function(i) {
           arr1 <- df1$result[[i]][[1]]
           arr2 <- df2$result[[i]][[1]]
           c(apply(arr1 - arr2, c(1, length(dim(arr1))), function(x) mean(abs(x))))
         })
  corr <- lapply(seq_len(nrow(df1)),
                  function(i) {
                    arr1 <- df1$result[[i]][[1]]
                    arr2 <- df2$result[[i]][[1]]
                    unlist(lapply(seq_len(dim(arr1)[1]), function(j) {
                      unlist(lapply(seq_len(rev(dim(arr1))[1]), function(k) {
                        dims  <- c("1", as.character(length(dim(arr1))))
                        m1 <- R.utils::extract(arr1, j, k, dims = dims)
                        m2 <- R.utils::extract(arr2, j, k, dims = dims)

                        if (all(m1 == m2)) {
                          res <- 1
                        } else {
                          res <- cor(m1, m2)
                        }

                        res
                      }))
                    }))
                  })

  freq <- unlist(lapply(error, length))
  res_comp <- res_comp[rep(seq_len(nrow(res_comp)), freq), ]
  res_comp$error <- unlist(error)
  res_comp$cor <- unlist(corr)

  list(res_comp, res_time)
}


load_conda_envs <- function(pkg, n_cpu = 1) {
  capture.output(
    capture.output(
      {
        Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")
        if (pkg %in% c("zennit", "captum")) {
          reticulate::use_condaenv("JSS_innsight_pytorch")
          import_torch <- reticulate::import("torch")
          import_torch$set_num_threads(as.integer(n_cpu))
          import_torch$set_num_interop_threads(as.integer(n_cpu))
        } else if (pkg == "deeplift") {
          Sys.setenv("RETICULATE_PYTHON" = reticulate::conda_python("JSS_innsight_tf_1"))
          reticulate::use_condaenv("JSS_innsight_tf_1")
          reticulate::py_capture_output({
            keras::use_condaenv("JSS_innsight_tf_1")
            tensorflow::tf$set_random_seed(42)
            library(keras)
          })
        } else if (pkg %in% c("innvestigate", "innsight_keras")) {
          Sys.setenv("RETICULATE_PYTHON" = reticulate::conda_python("JSS_innsight_tf_2"))
          reticulate::use_condaenv("JSS_innsight_tf_2")
          reticulate::py_capture_output({
            keras::use_condaenv("JSS_innsight_tf_2")
            tensorflow::tf$random$set_seed(42)
            library(keras)
          })
          library(torch)
          torch_set_num_threads(as.integer(n_cpu))
          torch_set_num_interop_threads(as.integer(n_cpu))
          library(innsight)
        } else {
          library(torch)
          torch_set_num_threads(as.integer(n_cpu))
          torch_set_num_interop_threads(as.integer(n_cpu))
          library(innsight)
        }
      }, type = "message"
    ), type = "output")
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")
}
