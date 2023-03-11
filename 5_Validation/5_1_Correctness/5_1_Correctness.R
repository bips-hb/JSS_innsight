
################################################################################
#
#   This script creates the figures from Section 5.1 ("Validation")
#
################################################################################

# Load required packages
library(innsight)
library(torch)
library(keras)
library(callr)
library(data.table)
library(R.utils)
library(cli)
library(ggplot2)
library(scales)
library(ggsci)
library(latex2exp)

# Keras and torch have to be installed properly
if (!is_keras_available()) {
  stop("Install Keras/TensorFlow via 'keras::install_keras()'")
}
if (!torch_is_installed()) {
  stop("Install libTorch via 'torch::install_torch()'")
}

################################################################################
#---------------------- Configuration and pre-processing -----------------------
################################################################################

# Global settings
num_models <- 50
num_outputs <- c(1, 5)
src_dir <- "5_Validation/5_1_Correctness"
batch_size <- 32
show <- TRUE

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
source("5_Validation/utils/preprocess/preprocess.R")
model_config <- preprocess(num_models, config, src_dir)

################################################################################
#------------------------------ Run benchmark ----------------------------------
################################################################################

source("5_Validation/utils/methods/benchmark.R")
result <- benchmark(methods, model_config, show, src_dir)

################################################################################
# ------------------------------- Create plots ---------------------------------
################################################################################

cli::cli_h2("Creating Plots")
source("5_Validation/utils/utils.R")

# result with the MAE error
res_error <- result[[1]]

if (!dir.exists(paste0(src_dir, "/figures"))) {
  dir.create(paste0(src_dir, "/figures"))
}

# Gradient-based methods
p <- ggplot(res_error[res_error$method_grp %in% c("Gradient-based"), ]) +
  geom_violin(aes(y = error, x = pkg, fill = method), scale = "width") +
  facet_wrap(vars(method_grp), scales = "free_x", ncol = 2,
             strip.position = "top") +
  scale_y_continuous(trans = log10_with_0_trans(9), limits = c(0, 1e0)) +
  add_gray_box() +
  scale_fill_manual(values = pal_npg(c("nrc"), 1)(7)[1:2],
                    labels = c("Gradient", TeX("Gradient$\\times$Input"))) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  ylab("Mean absolute error (MAE)") + xlab("Package") + theme_bw() +
  labs(fill = NULL) + theme(legend.position="top")
ggsave(paste0(src_dir, "/figures/mae_gradient_based.pdf"), p, width = 5, height = 5)

# DeepLift
p <- ggplot(res_error[res_error$method_grp %in% c("DeepLift"), ]) +
  geom_violin(aes(y = error, x = pkg, fill = method),) +
  facet_wrap(vars(method_grp), scales = "free_x", ncol = 2,
             strip.position = "top") +
  scale_y_continuous(trans = log10_with_0_trans(9), limits = c(0, 1e0)) +
  add_gray_box() +
  scale_fill_manual(values = pal_npg(c("nrc"), 1)(7)[3:4],
                    labels = c("Rescale", "RevealCancel")) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  ylab(NULL) + xlab("Package") + theme_bw() +
  labs(fill = NULL) +
  theme(legend.position="top")
ggsave(paste0(src_dir, "/figures/mae_deeplift.pdf"), p, width = 5, height = 5)

# LRP
p <- ggplot(res_error[res_error$method_grp %in% c("LRP"), ]) +
  geom_violin(aes(y = error, x = pkg, fill = method), scale = "width") +
  facet_grid(cols = vars(method_grp), rows = vars(bias), scales = "free_x") +
  scale_y_continuous(trans = log10_with_0_trans(9), limits = c(0, 1e0)) +
  add_gray_box() +
  scale_fill_manual(values = pal_npg(c("nrc"), 1)(7)[5:7],
                    labels = c("simple rule", TeX("$\\epsilon$-rule"),
                               TeX("$\\alpha$-$\\beta$-rule"))) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  ylab(NULL) + xlab("Package") + theme_bw() +
  labs(fill = NULL) +
  theme(legend.position="top")
ggsave(paste0(src_dir, "/figures/mae_lrp.pdf"), p, width = 5, height = 5)
