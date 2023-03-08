
################################################################################
#
#   This script creates the figures from the appendix ("Appendix 1")
#
################################################################################

#---------------------- Configuration and pre-processing -----------------------

# Define attributes
num_models <- 20
num_outputs <- c(1, 5)
src_dir <- "Appendix_1"
generate_models <- TRUE
run_benchmark <- TRUE
batch_size <- 50
show <- TRUE # only for debugging

# Define experiment configs
# models for tabular data
# (there is no batch normalization layer for tabular data)
config_tab = expand.grid()

# models for image data
config_2D <- expand.grid(
  input_shape = list(c(batch_size, 10, 10, 3)),
  bias = c(TRUE, FALSE),
  act = c("relu", "tanh"),
  batchnorm = c("before_act", "after_act"),
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
if (generate_models) {
  source("utils/preprocess/preprocess.R")
  model_config <- preprocess(num_models, config, src_dir)
} else {
  cli_progress_step("Loading model configuration file")
  path <- paste0(src_dir, "/model_config.rds")
  if (file.exists(path)) {
    model_config <- readRDS(path)
  } else {
    cli::cli_abort("Couldn't find file {.code model_config.rds} in folder ./{src_dir}/!")
  }
}

#-------------------------- Create benchmark results ---------------------------

# Start benchmark
if (run_benchmark) {
  source("utils/methods/benchmark.R")
  result <- benchmark(methods, model_config, show, src_dir)
} else {
  cli_progress_step("Loading results")
  path <- paste0(src_dir, "/results.rds")
  if (file.exists(path)) {
    result <- readRDS(path)
  } else {
    cli::cli_abort("Couldn't find file {.code results.rds} in folder ./{src_dir}/!")
  }
}

# ------------------------------- Create plots ---------------------------------
cli::cli_h2("Creating Plots")
library(ggplot2)
library(scales)
source("utils/utils.R")

# result with the MAE error
res_error <- result[[1]]
# result with the time comparison
res_time <- result[[2]]

if (!dir.exists(paste0(src_dir, "/figures"))) {
  dir.create(paste0(src_dir, "/figures"))
}

# Gradient-based methods
p <- ggplot(res_error[res_error$method_grp %in% c("Gradient", "SmoothGrad"), ]) +
  geom_violin(aes(y = error, x = method), fill = "darkgray", scale = "width") +
  facet_grid(cols = vars(pkg), rows = vars(bias), scales = "free_x") +
  scale_y_continuous(trans = log10_with_0_trans(10), limits = c(0, 1e0)) +
  add_gray_box() +
  geom_hline(yintercept = 0, alpha = 0.5) +
  labs(fill = "Method") +
  ylab("Mean Absolute Error (MAE)") + xlab("") + theme_bw()
ggsave(paste0(src_dir, "/figures/mae_gradient_based.pdf"), p, width = 10, height = 6)
ggsave(paste0(src_dir, "/figures/mae_gradient_based.png"), p, width = 10, height = 6, dpi = 300)

# LRP
p <- ggplot(res_error[res_error$method_grp == "LRP", ]) +
  geom_violin(aes(y = error, x = method, fill = bias), scale = "width") +
  facet_grid(cols = vars(pkg), scales = "free_x") +
  scale_y_continuous(trans = log10_with_0_trans(10), limits = c(0, 1e0)) +
  add_gray_box() +
  ggsci::scale_color_npg() +
  geom_hline(yintercept = 0, alpha = 0.5) +
  labs(fill = "Bias vector") +
  ylab("Mean Absolute Error (MAE)") + xlab("") + theme_bw()
ggsave(paste0(src_dir, "/figures/mae_lrp.pdf"), p, width = 12, height = 6)
ggsave(paste0(src_dir, "/figures/mae_lrp.png"), p, width = 12, height = 6, dpi = 300)

# DeepLift
df <- res_error[res_error$method_grp == "DeepLift", ]
df$batchnorm <- ifelse(df$batchnorm == "before_act", "BN before activation", "BN after activation")
df$pooling <- as.character(df$pooling)
df$pooling[df$pooling == "none"] <- "no pooling"
df$pooling[df$pooling == "avg"] <- "avg. pooling"
df$pooling[df$pooling == "max"] <- "max. pooling"
p <- ggplot(df) +
  geom_violin(aes(y = error, x = method, fill = pooling), scale = "width") +
  facet_grid(cols = vars(pkg), rows = vars(batchnorm), scales = "free_x") +
  scale_y_continuous(trans = log10_with_0_trans(10), limits = c(0, 1e0)) +
  add_gray_box() +
  geom_hline(yintercept = 0, alpha = 0.5) +
  labs(fill = "Pooling") +
  ylab("Mean Absolute Error (MAE)") + xlab("") + theme_bw()
ggsave(paste0(src_dir, "/figures/mae_deeplift.pdf"), p, width = 10, height = 6)
ggsave(paste0(src_dir, "/figures/mae_deeplift.png"), p, width = 10, height = 6, dpi = 300)


# Correlation
p <- ggplot(res_error[res_error$pkg %in% c("innvestigate") & res_error$method_grp %in% c("LRP"), ]) +
  geom_violin(aes(y = cor, x = method, fill = bias), scale = "width") +
  facet_grid(cols = vars(pkg), scales = "free") +
  ylab("Mean Absolute Error (MAE)") + xlab("") + theme_bw() + labs(fill = "Bias vector")
ggsave(paste0(src_dir, "/figures/cor_lrp.pdf"), p, width = 10, height = 6)
ggsave(paste0(src_dir, "/figures/cor_lrp.png"), p, width = 10, height = 6, dpi = 300)

p <- ggplot(df) +
  geom_violin(aes(y = cor, x = method, fill = pooling), scale = "width") +
  facet_grid(cols = vars(pkg), rows = vars(batchnorm), scales = "free") +
  ylab("mean absolute error (MAE)") + xlab("") + theme_bw()
ggsave(paste0(src_dir, "/figures/cor_deeplift.pdf"), p, width = 10, height = 6)
ggsave(paste0(src_dir, "/figures/cor_deeplift.png"), p, width = 10, height = 6, dpi = 300)
