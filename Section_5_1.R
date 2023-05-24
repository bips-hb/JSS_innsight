
################################################################################
#
#   This script creates the figures from Section 5.1 ("Validation")
#                   FIGURE 10 (a), (b) & (c)
#
################################################################################

# NOTE:
# For a minimal test execution in significantly less time, the number of models
# per architecture can be reduced, i.e., set `num_models <- 1` for the global
# attributes

# Minimal execution (takes only a couple of minutes)
num_models <- 2

# Paper execution (takes 1-2 days)
# num_models <- 50


# Load required packages
library("innsight")
library("torch")
library("keras")
library("callr")
library("data.table")
library("R.utils")
library("cli")
library("scales")
library("ggsci")

# Load LaTeX font (Latin modern), only relevant for setting the fonts as in the
# paper, but requires the latinmodern-math font
library("showtext")
font_add("LModern_math", "additional_files/latinmodern-math.otf")
showtext_auto()

library("ggplot2")


################################################################################
#                  Check if conda environments are installed
################################################################################
library("reticulate")

required_conda_envs <- c(
  "JSS_innsight_tf_1",   # for the python package 'deeplift'
  "JSS_innsight_tf_2",   # for the python package 'innvestigate'
  "JSS_innsight_pytorch" # for the python packages 'captum' and 'zennit'
)

if (any(!(required_conda_envs %in% conda_list()$name))) {
  x <- readline(paste0("Not all necessary conda environments are installed. Should",
                " I install them? (y/n)"))
  if (x == "y") {
    source("utils/create_condaenvs.R")

  } else {
    stop("Without the conda environments, the code cannot be executed!")
  }
}

################################################################################
#                       Configuration and pre-processing
################################################################################
start_time <- Sys.time()

# Global settings
num_outputs <- c(1, 5)
src_dir <- "tmp_results/5_1_Correctness"
batch_size <- 32
show <- FALSE # only for debugging

# Define experiment configs
# models for tabular data
config_tab = expand.grid(
  input_shape = list(c(batch_size, 10)),
  bias = c(TRUE, FALSE),
  act = c("relu", "tanh"),
  num_outputs = num_outputs
)

# models for image data
config_2D <- expand.grid(
  input_shape = list(c(batch_size, 10, 10, 3)),
  bias = c(TRUE, FALSE),
  act = c("relu", "tanh"),
  batchnorm = c("none"),
  pooling = c("none", "avg", "max"),
  num_outputs = num_outputs
)

config <- list(config_tab = config_tab, config_2D = config_2D)

# Define methods to be applied
methods <- c(
  "Gradient", "GradxInput",
  "LRP_simple_0", "LRP_epsilon_0.01", "LRP_alpha-beta_1", "LRP_alpha-beta_2",
  "DeepLift_rescale_zeros", "DeepLift_rescale_norm",
  "DeepLift_reveal-cancel_zeros", "DeepLift_reveal-cancel_norm"
)

# Generate models
source("utils/utils_validation/preprocess/preprocess.R")
model_config <- preprocess(num_models, config, src_dir)

################################################################################
#                               Run benchmark
################################################################################

source("utils/utils_validation/methods/benchmark.R")
result <- benchmark(methods, model_config, show, src_dir)

################################################################################
#                                 Create plots
################################################################################

cli::cli_h1("Creating Plots")
source("utils/utils_validation/utils.R")

# result with the MAE error
res_error <- result[[1]]

# Gradient-based methods (FIGURE 10 (a))
p <- ggplot(res_error[res_error$method_grp %in% c("Gradient-based"), ]) +
  geom_boxplot(aes(y = error, x = pkg, fill = method),
               outlier.size = 0.75,
               outlier.alpha = 0.25) +
  facet_grid(cols = vars(method_grp)) +
  scale_y_continuous(trans = log10_with_0_trans(9), limits = c(0, 1e0)) +
  add_gray_box() +
  scale_fill_manual(values = pal_npg(c("nrc"), 1)(7)[1:2],
                    labels = c("Gradient", "Gradient\u00D7Input")) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  labs(y = "Mean absolute error (MAE)",
       x = "Package",
       fill = NULL) +
  theme_bw() +
  theme(
    legend.position = "top",
    legend.spacing.x = unit(8, 'pt'),
    text = element_text(family = "LModern_math", size = 15))
ggsave("figures/FIGURE_10_a.pdf", p, width = 5, height = 5)

# DeepLift (FIGURE 10 (b))
p <- ggplot(res_error[res_error$method_grp %in% c("DeepLift"), ]) +
  geom_boxplot(aes(y = error, x = pkg, fill = method),
               outlier.size = 0.75,
               outlier.alpha = 0.25) +
  facet_grid(cols = vars(method_grp)) +
  scale_y_continuous(trans = log10_with_0_trans(9), limits = c(0, 1e0)) +
  add_gray_box() +
  scale_fill_manual(values = pal_npg(c("nrc"), 1)(7)[3:4],
                    labels = c("Rescale", "RevealCancel")) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  labs(y = NULL, x = "Package", fill = NULL) +
  theme_bw() +
  theme(legend.position="top",
        legend.spacing.x = unit(8, 'pt'),
        text = element_text(family="LModern_math", size = 15))
ggsave("figures/FIGURE_10_b.pdf", p, width = 5, height = 5)

# LRP (FIGURE 10 (c))
p <- ggplot(res_error[res_error$method_grp %in% c("LRP"), ]) +
  geom_boxplot(aes(y = error, x = pkg, fill = method),
               outlier.size = 0.75,
               outlier.alpha = 0.25) +
  facet_grid(cols = vars(method_grp), rows = vars(bias)) +
  scale_y_continuous(trans = log10_with_0_trans(9, 4), limits = c(0, 1e2)) +
  add_gray_box() +
  geom_hline(yintercept = 0, alpha = 0.5) +
  labs(y = NULL, x = "Package", fill = NULL) +
  theme_bw() +
  theme(legend.position="top",
        legend.spacing.x = unit(8, 'pt'),
        text = element_text(family="LModern_math", size = 15))+
  scale_fill_manual(values = pal_npg(c("nrc"), 1)(7)[5:7],
                    labels = c("simple rule", expression(epsilon*"-rule"),
                               expression(alpha*"-"*beta*"-rule")))
ggsave("figures/FIGURE_10_c.pdf", p, width = 5, height = 5)

################################################################################
#                             Outlier analysis
################################################################################

cli::cli_h1("Outlier Analysis")

# DeepLift
outliers_deeplift <- res_error[res_error$method_grp == "DeepLift" & res_error$error > 1e-6]
cli_text("Number (Percentage) of models with hyperbolic tangent as activation ",
      "among all DeepLift simulations exceeding an error of 1e-6: \n",
      sum(outliers_deeplift$act == "tanh"), " (",
      round(mean(outliers_deeplift$act == "tanh") * 100, 2), "%)")

# LRP
outliers_lrp <- res_error[res_error$method_grp == "LRP" &
                            res_error$method != "LRP (alpha-beta)" &
                            res_error$error > 1e-6]
cli_text("Number (Percentage) of models with hyperbolic tangent as activation ",
         "among all LRP simulations exceeding an error of 1e-6: \n",
         sum(outliers_lrp$act == "tanh"), " (",
         round(mean(outliers_lrp$act == "tanh") * 100, 2), "%)")

###############################################################################
#                             Print sessionInfo()
###############################################################################
cli_h1("Session Info")

sessionInfo()

time_diff <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("\nTotal execution time: ", col_blue(time_diff), " mins\n")
