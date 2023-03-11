
################################################################################
#
#   This script creates the figures from Section 5.2 ("Validation")
#
################################################################################

# Load required packages
library(innsight)
library(torch)
library(keras)
library(callr)
library(cowplot)
library(data.table)
library(R.utils)
library(cli)
library(ggplot2)
library(scales)
library(ggsci)
library(latex2exp)
source("5_Validation/utils/utils_time.R")

# Keras and torch have to be installed properly
if (!is_keras_available()) {
  stop("Install Keras/TensorFlow via 'keras::install_keras()'")
}
if (!torch_is_installed()) {
  stop("Install libTorch via 'torch::install_torch()'")
}

################################################################################
#-------------------- Time comparison for output nodes -------------------------
################################################################################
num_outputs <- c(1, seq(5, 50, by = 5))
num_hidden_layers <- c(2)
num_hidden_units <- c(64)
num_inputs <- c(10)
num_models <- 20
batch_sizes <- c(32)
src_dir <- "5_Validation/5_2_Time/Num_Outputs"

res <- compare_time(num_models, num_outputs, num_inputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)
create_plots(res, "num_outputs", "5_Validation/5_2_Time")

################################################################################
#-------------------- Time comparison for number layers ------------------------
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(1, seq(5, 50, by = 5))
num_hidden_units <- c(32)
num_inputs <- c(10)
num_models <- 20
batch_sizes <- c(32)
src_dir <- "5_Validation/5_2_Time/Num_Layers"

res <- compare_time(num_models, num_outputs, num_inputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)
create_plots(res, "hidden_depth", "5_Validation/5_2_Time")

################################################################################
#-------------------- Time comparison for hidden width -------------------------
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- c(10, seq(100, 1000, by = 100))
num_inputs <- c(10)
num_models <- 20
batch_sizes <- c(10)
src_dir <- "5_Validation/5_2_Time/Num_Units"

res <- compare_time(num_models, num_outputs, num_inputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)
create_plots(res, "hidden_width", "5_Validation/5_2_Time")

################################################################################
#-------------------- Time comparison for batch_size ---------------------------
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- c(32)
num_inputs <- c(10)
num_models <- 20
batch_sizes <- c(10, seq(100, 1000, by = 100))
src_dir <- "5_Validation/5_2_Time/Batch_Size"

res <- compare_time(num_models, num_outputs, num_inputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)

create_plots(res, "batch_size", "5_Validation/5_2_Time")
