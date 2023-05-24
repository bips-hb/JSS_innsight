
################################################################################
#
#   This script creates the figures from Section 5.2 ("Validation")
#                   Figure 12 (a) & (b)
#   and the additional figures from Appendix A.3
#               Figure 14, 15, 16, 17 & 18
#
################################################################################

# NOTE:
# For a minimal test execution in significantly less time, the number of models
# per architecture can be reduced, i.e. set `num_models <- 1` for the global
# attributes. In addition, the step size of the varying parameter can also be
# lowered, e.g., for the number of hidden layers, set `c(2, 25, 50)` instead
# of `c(2, seq(5, 50, by = 5))`.

# Minimal execution
num_models <- 1
IMAGE_SHAPES <- lapply(c(10, seq(5) * 80 + 4), function(x) c(x,x,3))
NUM_OUTPUTS <- c(1, seq(20, 100, by = 20))
NUM_HIDDEN_LAYERS <- c(2, seq(10, 50, by = 10))
NUM_HIDDEN_UNITS <- list(c(100, seq(600, 3000, by = 600)), c(10, seq(200, 1000, by = 200)))
BATCH_SIZES <- c(100, seq(600, 3000, by = 600))

# Paper execution
#num_models <- 20
#IMAGE_SHAPES <- lapply(c(10, seq(10) * 40 + 4), function(x) c(x,x,3))
#NUM_OUTPUTS <- c(1, seq(10, 100, by = 10))
#NUM_HIDDEN_LAYERS <- c(2, seq(5, 50, by = 5))
#NUM_HIDDEN_UNITS <- list(c(100, seq(300, 3000, by = 300)), c(10, seq(100, 1000, by = 100)))
#BATCH_SIZES <- c(100, seq(300, 3000, by = 300))

# Load required packages
library("innsight")
library("torch")
library("keras")
library("callr")
library("cowplot")
library("data.table")
library("R.utils")
library("cli")
library("scales")
library("ggsci")
library("cowplot")

# Load LaTeX font (Latin modern), only relevant for setting the fonts as in the
# paper, but requires the latinmodern-math font
library("showtext")
font_add("LModern_math", "additional_files/latinmodern-math.otf")
showtext_auto()

library("ggplot2")


# Load utility functions
source("utils/utils_validation/utils_time.R")


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

start_time <- Sys.time()

################################################################################
#                 Time comparison for image size (FIGURE 18)
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- list(c(16), c(5)) # first for tabular (units) and second for image (filters)
image_shapes <- IMAGE_SHAPES
batch_sizes <- c(16)
src_dir <- "tmp_results/5_2_Image_size"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes,
                    image_shapes = image_shapes)
create_plots(res, "num_inputs")

################################################################################
#           Time comparison for output nodes (FIGURE 12 (b) & 14)
################################################################################
num_outputs <- NUM_OUTPUTS
num_hidden_layers <- c(2)
num_hidden_units <- list(c(16), c(5)) # first for tabular (units) and second for image (filters)
batch_sizes <- c(16)
src_dir <- "tmp_results/5_2_Num_Outputs"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)
create_plots(res, "num_outputs")
create_figure_plot(res, "num_outputs")

################################################################################
#            Time comparison for number layers (FIGURE 12 (a) & 15)
################################################################################
num_outputs <- c(1)
num_hidden_layers <- NUM_HIDDEN_LAYERS
num_hidden_units <-  list(c(16), c(5))
batch_sizes <- c(16)
src_dir <- "tmp_results/5_2_Num_Layers"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)
create_plots(res, "hidden_depth")
create_figure_plot(res, "hidden_depth")

################################################################################
#               Time comparison for hidden width (FIGURE 16)
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- NUM_HIDDEN_UNITS
batch_sizes <- c(16)
src_dir <- "tmp_results/5_2_Num_Units"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)
create_plots(res, "hidden_width")

################################################################################
#               Time comparison for batch_size (FIGURE 17)
################################################################################
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- list(c(16), c(5))
batch_sizes <- BATCH_SIZES
src_dir <- "tmp_results/5_2_Batch_Size"

res <- compare_time(num_models, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes)

create_plots(res, "batch_size")


###############################################################################
#                             Print sessionInfo()
###############################################################################
cli_h1("Session Info")

sessionInfo()

time_diff <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("\nTotal execution time: ", col_blue(time_diff), " mins\n")
