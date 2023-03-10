
################################################################################
#
#   This script creates the figures from the appendix ("Appendix 1")
#
################################################################################

#---------------------- Configuration and pre-processing -----------------------

# Define attributes
num_models <- 20
num_outputs <- c(1, 5)
src_dir <- "Appendix_A"
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
library(ggsci)
library(latex2exp)
source("utils/utils.R")

# result with the MAE error
res_error <- result[[1]]
# result with the time comparison
res_time <- result[[2]]

if (!dir.exists(paste0(src_dir, "/figures"))) {
  dir.create(paste0(src_dir, "/figures"))
}

# Gradient-based methods
p <- ggplot(res_error[res_error$method_grp == "Gradient-based", ]) +
  geom_violin(aes(y = error, x = pkg, fill = method), scale = "width") +
  facet_wrap(vars(method_grp), scales = "free_x", ncol = 2, strip.position = "top") +
  scale_y_continuous(trans = log10_with_0_trans(9), limits = c(0, 1e0)) +
  add_gray_box() +
  scale_fill_manual(values = pal_npg(c("nrc"), 1)(7)[1:2], labels = c("Gradient", TeX("Gradient $\\times$ Input"))) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  ylab("Mean Absolute Error (MAE)") + xlab("Package") + theme_bw() +
  labs(fill = NULL) + theme(legend.position="top")
ggsave(paste0(src_dir, "/figures/mae_gradient_based_with_bn.pdf"), p, width = 4, height = 5)

# DeepLift captum
df <- res_error[res_error$method_grp == "DeepLift" & res_error$pkg == "captum", ]
df$pooling <- as.character(df$pooling)
df$pooling[df$pooling %in% c("none", "avg")] <- "no or avg. pooling"
df$pooling[df$pooling == "max"] <- "max. pooling"
p <- ggplot(df) +
  geom_violin(aes(y = error, x = pkg, fill = method),) +
  facet_grid(cols = vars(method_grp, batchnorm), rows = vars(pooling, act), scales = "free_x") +
  scale_y_continuous(trans = log10_with_0_trans(9), limits = c(0, 1e0)) +
  add_gray_box() +
  scale_fill_manual(values = pal_npg(c("nrc"), 1)(7)[3:4], labels = c("Rescale", "RevealCancel")) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  ylab(NULL) + xlab("Package") + theme_bw() +
  labs(fill = NULL) +
  theme(legend.position="top")
ggsave(paste0(src_dir, "/figures/mae_deeplift_captum_with_bn.pdf"), p, width = 4, height = 5)

# DeepLift deeplift
df <- res_error[res_error$method_grp == "DeepLift" & res_error$pkg == "deeplift", ]
df$batchnorm <- ifelse(df$batchnorm == "before_act", "BN before activation", "BN after activation")
p <- ggplot(df) +
  geom_violin(aes(y = error, x = pkg, fill = method),) +
  facet_grid(cols = vars(method_grp), rows = vars(batchnorm), scales = "free_x") +
  scale_y_continuous(trans = log10_with_0_trans(9), limits = c(0, 1e0)) +
  add_gray_box() +
  scale_fill_manual(values = pal_npg(c("nrc"), 1)(7)[3:4], labels = c("Rescale", "RevealCancel")) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  ylab(NULL) + xlab("Package") + theme_bw() +
  labs(fill = NULL) +
  theme(legend.position="top")
ggsave(paste0(src_dir, "/figures/mae_deeplift_deeplift_with_bn.pdf"), p, width = 4, height = 5)

# LRP
p <- ggplot(res_error[res_error$method_grp == "LRP", ]) +
  geom_violin(aes(y = error, x = pkg, fill = method), scale = "width") +
  facet_grid(cols = vars(method_grp), rows = vars(bias), scales = "free_x") +
  scale_y_continuous(trans = log10_with_0_trans(9), limits = c(0, 1e0)) +
  add_gray_box() +
  scale_fill_manual(values = pal_npg(c("nrc"), 1)(7)[5:7],
                    labels = c("simple rule", TeX("$\\epsilon$-rule"), TeX("$\\alpha$-$\\beta$-rule"))) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  ylab(NULL) + xlab("Package") + theme_bw() +
  labs(fill = NULL) +
  theme(legend.position="top")
ggsave(paste0(src_dir, "/figures/mae_lrp_with_bn.pdf"), p, width = 4, height = 5)


ggplot(res_error) +
  geom_violin(aes(y = cor, x = pkg, fill = method)) +
  facet_grid(cols = vars(method_grp), rows = vars(pooling), scales = "free")

res_time$method_grp <- factor(res_time$method_grp, levels = c("Gradient-based", "DeepLift", "LRP"))
res_time$method <- factor(res_time$method,
                          levels = unique(res_time$method)[c(1,2,6,7,3,4,5)])
ggplot(res_time) +
  geom_violin(aes(y = time_eval, x = pkg, fill = method), scale = "width") +
  geom_hline(yintercept = 0) +
  facet_grid(cols = vars(method_grp), scales = "free_x") +
  scale_fill_manual(values = pal_npg(c("nrc"), 1)(7),
                   labels = c("Gradient", TeX("Gradient $\\times$ Input"),
                              "rescale", "reveal-cancel",
                              "simple rule", TeX("$\\epsilon$-rule"), TeX("$\\alpha$-$\\beta$-rule"))) +
  ylim(c(-0.5, 0.25)) + labs(fill = NULL) +
  guides(fill = guide_legend(nrow = 1)) +
  theme_bw() +
  theme(legend.position="top")



