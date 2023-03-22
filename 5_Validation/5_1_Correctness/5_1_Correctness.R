
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
library(scales)
library(ggsci)

# Load LaTeX font (Latin modern)
library(showtext)
font_add("LModern_math", "/home/koenen/fonts/latinmodern-math.otf")
showtext_auto()
library(ggplot2)


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
ggsave(paste0(src_dir, "/figures/mae_gradient_based.pdf"), p, width = 5, height = 5)
ggsave(paste0(src_dir, "/figures/mae_gradient_based.png"),
       p + theme(text = element_text(size = 50)), width = 5, height = 5, dpi = 300)

# DeepLift
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
ggsave(paste0(src_dir, "/figures/mae_deeplift.pdf"), p, width = 5, height = 5)
ggsave(paste0(src_dir, "/figures/mae_deeplift.png"),
       p + theme(text = element_text(size = 50)), width = 5, height = 5, dpi = 300)

# LRP
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
ggsave(paste0(src_dir, "/figures/mae_lrp.pdf"), p, width = 5, height = 5)
ggsave(paste0(src_dir, "/figures/mae_lrp.png"),
       p + theme(text = element_text(size = 50)), width = 5, height = 5, dpi = 300)
