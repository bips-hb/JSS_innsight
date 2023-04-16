
################################################################################
#
#   This script creates the figures from Section 5.2 ("Validation")
#
################################################################################

# NOTE:
# For a minimal test execution in significantly less time, the number of models
# per architecture can be reduced, i.e. set `num_models <- 1` for the global
# attributes. In addition, the step size of the varying parameter can also be
# lowered, e.g., for the number of hidden layers, set `c(2, 25, 50)` instead
# of `c(2, seq(5, 50, by = 5))`.


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


# Load LaTeX font (Latin modern), only relevant for setting the fonts as in the
# paper, but requires the latinmodern-math font
use_paper_font <- FALSE
if (use_paper_font) {
  library(showtext)
  font_add("LModern_math", "/home/koenen/fonts/latinmodern-math.otf")
  showtext_auto()
  font_name <- "LModern_math"
} else {
  font_name <- NULL
}
library(ggplot2)


source("5_Validation/utils/utils_time.R")

num_models <- 20 # <<<<<<<<<<<<<<<<<<<<<< Reduce this value for a test execution
num_cpus <- 1


################################################################################
#-------------------- Time comparison for image size ---------------------------
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- list(c(16), c(5)) # first for tabular (units) and second for image (filters)
image_shapes <- lapply(c(10, seq(10) * 40 + 4), function(x) c(x,x,3)) # <<<<<<<<<<<<<<<<<<<<<< Reduce this value for a test execution
batch_sizes <- c(16)
src_dir <- "5_Validation/5_2_Time/Image_size"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes, num_cpus,
                    image_shapes = image_shapes)
create_plots(res, "num_inputs", "5_Validation/5_2_Time", font_name)

################################################################################
#-------------------- Time comparison for output nodes -------------------------
################################################################################
num_outputs <- c(1, seq(10, 100, by = 10)) # <<<<<<<<<<<<<<<<<<<<<< Reduce this value for a test execution
num_hidden_layers <- c(2)
num_hidden_units <- list(c(16), c(5)) # first for tabular (units) and second for image (filters)
batch_sizes <- c(16)
src_dir <- "5_Validation/5_2_Time/Num_Outputs"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes, num_cpus)
create_plots(res, "num_outputs", "5_Validation/5_2_Time", font_name)
create_figure_plot(res, "5_Validation/5_2_Time", "num_outputs", font_name)

################################################################################
#-------------------- Time comparison for number layers ------------------------
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(2, seq(5, 50, by = 5)) # <<<<<<<<<<<<<<<<<<<<<< Reduce this value for a test execution
num_hidden_units <-  list(c(16), c(5))
batch_sizes <- c(16)
src_dir <- "5_Validation/5_2_Time/Num_Layers"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes, num_cpus)
create_plots(res, "hidden_depth", "5_Validation/5_2_Time", font_name)
create_figure_plot(res, "5_Validation/5_2_Time", "hidden_depth", font_name)

################################################################################
#-------------------- Time comparison for hidden width -------------------------
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- list(c(100, seq(300, 3000, by = 300)), c(10, seq(100, 1000, by = 100))) # <<<<<<<<<<<<<<<<<<<<<< Reduce this value for a test execution
batch_sizes <- c(16)
src_dir <- "5_Validation/5_2_Time/Num_Units"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes, num_cpus)
create_plots(res, "hidden_width", "5_Validation/5_2_Time", font_name)

################################################################################
#-------------------- Time comparison for batch_size ---------------------------
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- list(c(16), c(5))
batch_sizes <- c(100, seq(300, 3000, by = 300)) # <<<<<<<<<<<<<<<<<<<<<< Reduce this value for a test execution
src_dir <- "5_Validation/5_2_Time/Batch_Size"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes, num_cpus)

create_plots(res, "batch_size", "5_Validation/5_2_Time", font_name)
