
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
library(scales)
library(ggsci)
library(cowplot)


# Load LaTeX font (Latin modern)
library(showtext)
font_add("LModern_math", "/home/koenen/fonts/latinmodern-math.otf")
showtext_auto()
library(ggplot2)


source("5_Validation/utils/utils_time.R")


################################################################################
#-------------------- Time comparison for output nodes -------------------------
################################################################################
num_outputs <- c(1, seq(10, 100, by = 10))
num_hidden_layers <- c(2)
num_hidden_units <- list(c(16), c(5)) # first for tabular (units) and second for image (filters)
num_models <- 30
batch_sizes <- c(16)
src_dir <- "5_Validation/5_2_Time/Num_Outputs"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)
create_plots(res, "num_outputs", "5_Validation/5_2_Time")

################################################################################
#-------------------- Time comparison for number layers ------------------------
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(1, seq(5, 50, by = 5))
num_hidden_units <-  list(c(16), c(5))
num_models <- 30
batch_sizes <- c(16)
src_dir <- "5_Validation/5_2_Time/Num_Layers"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)
create_plots(res, "hidden_depth", "5_Validation/5_2_Time")

################################################################################
#-------------------- Time comparison for hidden width -------------------------
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- list(c(10, seq(300, 3000, by = 300)), c(10, seq(100, 1000, by = 100)))
num_models <- 30
batch_sizes <- c(16)
src_dir <- "5_Validation/5_2_Time/Num_Units"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)
create_plots(res, "hidden_width", "5_Validation/5_2_Time")

################################################################################
#-------------------- Time comparison for batch_size ---------------------------
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- list(c(16), c(5))
num_models <- 30
batch_sizes <- c(10, seq(300, 3000, by = 300))
src_dir <- "5_Validation/5_2_Time/Batch_Size"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)

create_plots(res, "batch_size", "5_Validation/5_2_Time")
