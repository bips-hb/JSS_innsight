library("callr")
library("keras")
library("cli")

###############################################################################
#                           Utility functions
###############################################################################
func_innv <- function(input, model_path) {
  Sys.setenv("RETICULATE_PYTHON" = reticulate::conda_python("JSS_innsight_tf_2"))
  reticulate::use_condaenv("JSS_innsight_tf_2")
  keras::use_condaenv("JSS_innsight_tf_2")

  tf <- reticulate::import("tensorflow")
  innv <- reticulate::import("innvestigate")
  keras <- reticulate::import("keras")
  np <- reticulate::import("numpy")
  tf$compat$v1$disable_eager_execution()

  # Load model
  model <- keras$models$load_model(model_path)

  # epsilon rule (0.001)
  analyzer = innv$create_analyzer("lrp.epsilon", model, epsilon = 0.01)
  res1 <- apply(analyzer$analyze(input), 1, sum)

  # alpha-beta rule (alpha = 1)
  analyzer = innv$create_analyzer("lrp.alpha_beta", model, alpha = 1)
  res2 <- apply(analyzer$analyze(input), 1, sum)

  # alpha-beta rule (alpha = 2)
  analyzer = innv$create_analyzer("lrp.alpha_beta", model, alpha = 2)
  res3 <- apply(analyzer$analyze(input), 1, sum)

  # model output
  out <- model$predict(input)

  data.frame(epsilon_0.001 = c(res1), alpha_beta_1 = c(res2),
             alpha_beta_2 = c(res3), out = c(out))
}

func_innsight <- function(input, model_path) {
  library("innsight")
  library("torch")
  library("keras")
  model <- load_model_hdf5(model_path)
  conv <- Converter$new(model)
  lrp <- LRP$new(conv, input, channels_first = FALSE, verbose = FALSE,
                 rule_name = get_rule_name("epsilon"),
                 rule_param = get_rule_arg(0.01))
  res1 <- apply(innsight::get_result(lrp), c(1), sum)
  lrp <- LRP$new(conv, input, channels_first = FALSE, verbose = FALSE,
                 rule_name = get_rule_name("alpha_beta"),
                 rule_param = get_rule_arg(1))
  res2 <- apply(innsight::get_result(lrp), c(1), sum)
  lrp <- LRP$new(conv, input, channels_first = FALSE, verbose = FALSE,
                 rule_name = get_rule_name("alpha_beta"),
                 rule_param = get_rule_arg(2))
  res3 <- apply(innsight::get_result(lrp), c(1), sum)

  out <- as.array(conv$model(torch_tensor(input), channels_first = FALSE)[[1]])

  data.frame(epsilon_0.001 = c(res1), alpha_beta_1 = c(res2),
             alpha_beta_2 = c(res3), out = c(out))
}

get_rule_name <- function(rule_name) {
  list(
    Dense_Layer = rule_name,
    Conv2D_Layer = rule_name,
    AvgPool2D_Layer = "simple",
    BatchNorm_Layer = rule_name
  )
}

get_rule_arg <- function(rule_arg) {
  list(
    Dense_Layer = rule_arg,
    Conv2D_Layer = rule_arg,
    BatchNorm_Layer = rule_arg
  )
}

################################################################################
#                  Check if conda environments are installed
################################################################################
library("reticulate")

if (!"JSS_innsight_tf_2" %in% conda_list()$name) {
  x <- readline(paste0("Not all necessary conda environments are installed. Should",
                       " I install them? (y/n)"))
  if (x == "y") {
    conda_create(envname = "JSS_innsight_tf_2", python_version = "3.8.15")
    conda_install(envname = "JSS_innsight_tf_2",
                  packages = c("tensorflow-cpu==2.10", "keras==2.10.0",
                               "innvestigate==2.0.2"),
                  pip = TRUE)

  } else {
    stop("Without the conda environment, the code cannot be executed!")
  }
}


###############################################################################
#     Problem: LRP in innvestigate with bias in linear layers
###############################################################################
start_time <- Sys.time()

input <- layer_input(shape = c(1))
output <- input %>%
  layer_dense(1, use_bias = TRUE, weights = list(array(1, dim = c(1, 1)), array(-0.25)))
model <- keras_model(input, output)

model_path <- "tmp_results/Appendix_B/model_LRP_with_bias.h5"
model$save(model_path)

input <- array(1, dim = c(1, 1))
res_innv <- r(func_innv, args = list(input = input, model_path = model_path))
res_innsight <- func_innsight(input, model_path)

cli_h1("Results")
cli_h3("iNNvestigate")
res_innv
cli_h3("innsight")
res_innsight

###############################################################################
#                             Print sessionInfo()
###############################################################################
cli_h1("Session Info")

sessionInfo()

time_diff <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("\nTotal execution time: ", col_blue(time_diff), " mins\n")
