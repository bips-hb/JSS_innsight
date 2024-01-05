################################################################################
#
#               REPLICATION MATERIAL FOR THE JSS SUBMISSION
#     "Interpreting Deep Neural Networks with the R Package innsight"
#
################################################################################

# All code-based figures and results of the paper are generated with the
# following script and saved in the folder `figures/`.

################################################################################
#
#               GENERAL REQUIREMENTS FOR CODE EXECUTION
#
################################################################################

## Load packages ---------------------------------------------------------------
library("innsight")
library("torch")
library("keras")
library("palmerpenguins")
library("neuralnet")
library("ggplot2")
library("cli")
library("gridExtra")
library("callr")
library("data.table")
library("R.utils")
library("scales")
library("ggsci")
library("cowplot")
library("reticulate")
library("png")
library("grid")
library("showtext")


# The torch package must be installed correctly
# NOTE: If a GPU is available, torch installs the CUDA version by default, but
# this is not used for our package. To install only the CPU variant even if a
# GPU is present, set the CUDA environment variable to "cpu"
# (i.e. `Sys.setenv(CUDA = 'cpu')`) and install torch with `torch_install`. See
# `?torch_install` for details.
if (!torch::torch_is_installed()) stop("Call `torch::install_torch()`")

# Make sure TensorFlow is installed correctly
if (!is_keras_available()) stop("Call `keras::install_keras()`")

# Disable GPU usage (since it is not necessary for our examples and
# requires a more complicated installation including CUDA)
Sys.setenv(CUDA_VISIBLE_DEVICES = "")

# Create Conda environments ----------------------------------------------------
required_conda_envs <- c(
  "JSS_innsight_tf_1",   # for the python package 'deeplift'
  "JSS_innsight_tf_2",   # for the python package 'innvestigate'
  "JSS_innsight_pytorch" # for the python packages 'captum', 'zennit' and 'shap'
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
#
#                 CODE FOR SECTION 3 ("Bike-sharing dataset")
#                             FIGURE 5 (a)
#                       AND OUTPUT FOR APPENDIX B
#
################################################################################
start_time <- Sys.time()

# Set the seed for reproducibility
set.seed(42)

# Load the bike-sharing dataset and restrict the dataset to specific numerical
# or binary variables
bike <- read.csv("additional_files/bike_sharing/day.csv")
bike <- bike[, c("cnt", "holiday", "workingday", "temp", "hum", "windspeed")]

# Scale the outcome "cnt" to bikes per 10,000 and transform the whole dataset
# to a numerical matrix
bike$cnt <- bike$cnt / 10000
bike <- as.matrix(bike)

# Fit the neural network with one hidden layer and 64 nodes using
# the package "neuralnet"
model <- neuralnet(cnt~., data = bike, hidden = c(64), linear.output = TRUE)

# Convert the model and pass the output label
conv <- convert(model, output_names = c("Number of rented bikes/10,000"))

# Run DeepSHAP for the first 20 instances and use the whole dataset as the
# reference dataset (by default only 100 samples of this dataset are
# used for the calculation)
res_deepshap <- run_deepshap(conv, bike[1:20, -1], data_ref = bike[, -1])

# Show the results as an array, plot the results for the first data point
# and show the boxplots for over all 20 instances
head(get_result(res_deepshap))
plot(res_deepshap)
boxplot(res_deepshap, ref_data_idx = 1) +
  theme(text = element_text(face = "bold"))

# Combine and save the plot (FIGURE 5 (a))
p1 <- plot(res_deepshap)
p2 <- boxplot(res_deepshap, ref_data_idx = 1) +
  theme(text = element_text(face = "bold"))

p <- plot_grid(plot(p1), plot(p2), ncol = 1)
ggplot2::ggsave("figures/FIGURE_5_a.pdf", p, width = 6, height = 6)

# APPENDIX B: Model as a list
conv <- convert(model, output_names = c("Number of rented bikes/10,000"),
                save_model_as_list = TRUE)
str(conv$model_as_list, max.level = 3)

time_diff <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("Total execution time: ", col_blue(time_diff), " mins\n")

################################################################################
#
#                 CODE FOR SECTION 4.1 ("Penguin dataset")
#                     FIGURE 7 (a) AND FIGURE 7 (b)
#
################################################################################
start_time <- Sys.time()

# Set the seed for reproducibility
set.seed(42)

# Load the penguins dataset, remove any rows with missing values, and scale
# numerical variables
data <- na.omit(penguins[, c(1, 3, 4, 5, 6)])
data[, 2:5] <- scale(data[, 2:5])

# Split the data into training and testing sets and remove the variable
# 'species' from the test data as it isn't part of the mode input
train_idx <- sample.int(nrow(data), as.integer(nrow(data) * 0.75))
train_data <- data[train_idx, ]
test_data <- data[-train_idx, -1]

# Train a neural network model to predict species using the training data
model <- neuralnet(species ~ .,
                   data = train_data, hidden = 128, act.fct = "logistic",
                   err.fct = "ce", linear.output = FALSE)

# Convert the neural network model using the innsight package
conv <- convert(model)

# Apply the IntegratedGradient method with the test dataset's feature mean
# values as the reference value and 100 steps for the integral approximation.
intgrad <- run_intgrad(conv, test_data,
                       x_ref = matrix(colMeans(test_data), 1))

# Plot the relevance scores for two outputs of the neural network model
# This creates the FIGURE 7(a)
p <- plot(intgrad, data_idx = c(1, 50), output_label = c("Adelie", "Gentoo")) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6))
p

# Save figure
ggsave("figures/FIGURE_7_a.pdf", print(p), width = 6, height = 6)

# Create a box plot of the relevance scores for two outputs of the neural
# network model using the innsight package
# This creates the FIGURE 7 (b)
p <- boxplot(intgrad, output_label = c("Adelie", "Gentoo"), ref_data_idx = 1) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6))
p

# Save figure
ggsave("figures/FIGURE_7_b.pdf", print(p), width = 6, height = 6)

time_diff <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("Total execution time: ", col_blue(time_diff), " mins\n")


################################################################################
#
#                 CODE FOR SECTION 4.2 ("Melanoma dataset")
#                        FIGURE 9 (a) and 9 (b)
#
################################################################################
start_time <- Sys.time()

# Set model path
model_path <- "additional_files/melanoma_model.h5"

# Load the model
model <- load_model_tf(model_path, compile = FALSE)

# Define the input names and output name for the model
input_names <- list(
  list(paste0("C", 1:3), paste0("H", 1:224), paste0("W", 1:224)),
  list(c("Sex: Male", "Sex: Female", "Age",
         "Loc: Head/neck", "Loc: Torso", "Loc: Upper extrm.",
         "Loc: Lower extrem.", "Loc: Palms/soles", "Loc: Oral/genital",
         "Loc: Missing")))
output_name <- c("Probability of malignant lesion")

# Create a Converter object for the model
converter <- convert(model, input_names = input_names,
                     output_names = output_name)

# Get input data
image_files <- c("ISIC_6535558", "ISIC_7291021", "ISIC_0946787")

# Load images (this requires some helper methods)
source("utils/utils_melanoma.R")
images <- load_images(image_files)

# Load tabular inputs
df <- readRDS("additional_files/melanoma_tabular_data.rds")
# Sort the data instances
df <- df[match(image_files, df$image_name), ]
# Encode categorical features
tab_input <- encode_categorical(df)
tab_input <- as.matrix(tab_input[, 2:11])

# Create combined input
inputs <- list(images, tab_input)

# Add prediction values of the inputs
df$preds <- c(predict(model, inputs))

# Show the output
df

# Set the LRP rules and apply the method
rule_name <- list(Conv2D_Layer = "alpha_beta", Dense_Layer = "epsilon")
rule_param <- list(Conv2D_Layer = 1.5, Dense_Layer = 0.01)
res <- run_lrp(converter, inputs, channels_first = FALSE,
               rule_name = rule_name, rule_param = rule_param)

# Create the S4 'innsight_ggplot2' object (the suggested packages 'gridExtra'
# and 'gtable' are required for advanced plots)
require("gridExtra")
require("gtable")
p <- plot(res, data_idx = 1:3) + theme_bw()

# Modify the plot layout
# Change facet labels in the plot top left
p[1, 1] <- p[1, 1, restyle = FALSE] +
  facet_grid(cols = vars(model_input),
             labeller = as_labeller(c(Input_1 = "Image input")))
# Change the facet labels in the plot top right
p[1, 2] <- p[1, 2, restyle = FALSE] +
  facet_grid(cols = vars(model_input),
             rows = vars(data),
             labeller = as_labeller(c(Input_2 = "Tabular input",
                                      data_1 = "malignant")))
# Rotate the x-axis labels
p <- p +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6))


# Plot the modified plot (all additional arguments, e.g. 'heights' are
# forwarded to the method 'gridExtra::arrangeGrob()')
# This creates FIGURE 9 (b)
p <- plot(p, heights = c(0.31, 0.31, 0.38))

# Save plot
ggsave(filename="figures/FIGURE_9_b.pdf", plot=p,
       width = 30, height = 19, units = "cm")


# The following code creates the pngs of the input images, which are
# combined in the paper in LaTeX and shown as FIGURE 9 (a).
for (i in seq_len(dim(images)[[1]])) {
  png(file=
        paste0("figures/", df[i, ]$image_name, "_",
               df[i,]$benign_malignant,  ".png"),
      width=480, height=240, res=300)
  image <- images[i,,,, drop = FALSE]
  dim(image) <- c(224,224,3)

  # de-normalize the image
  image[,,1] <- image[,,1] + 0.80612123
  image[,,2] <- image[,,2] + 0.62106454
  image[,,3] <- image[,,3] + 0.591202
  par(mar=c(0,0,0,0))
  plot.new()
  rasterImage(image, xleft = -0.04, xright = 1.04,
              ytop = -0.04, ybottom = 1.04, interpolate = FALSE)
  dev.off()
}

# The following code creates a similar version of FIGURE 9 (a) (in the paper,
# the images are included individually)
plots <- lapply(paste0("figures/", image_files, "_", df$benign_malignant, ".png"), function(x) {
  img <- as.raster(readPNG(x))
  rasterGrob(img, interpolate = FALSE, vp = viewport(width = 1, height = 0.9))
})
ggsave("figures/FIGURE_9_a.pdf",
       marrangeGrob(grobs = plots, nrow=3, ncol=1, top=NULL))

time_diff <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("Total execution time: ", col_blue(time_diff), " mins\n")


################################################################################
#
#                 CODE FOR SECTION 5.1 ("Validity comparison")
#                         FIGURE 10 (a), (b) & (c)
#
################################################################################
start_time <- Sys.time()
set.seed(42)

# NOTE:
# For a minimal test execution in significantly less time, the number of models
# per architecture can be reduced, i.e., set `num_models <- 1` for the global
# attributes

# Minimal execution (takes only a couple of minutes)
num_models <- 1

# Paper execution (takes one day)
# num_models <- 50

# Load LaTeX font (Latin modern), only relevant for setting the fonts as in the
# paper, but requires the latinmodern-math font
font_add("LModern_math", "additional_files/latinmodern-math.otf")
showtext_auto()

# Configuration and pre-processing ---------------------------------------------

# Global settings
num_outputs <- c(1, 5)
src_dir <- "tmp_results/5_1_Correctness"
batch_size <- 16
num_refs <- 32
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
  input_shape = list(c(batch_size, 32, 32, 3)),
  bias = c(TRUE, FALSE),
  act = c("relu", "tanh"),
  batchnorm = c("none"),
  pooling = c("none", "avg", "max"),
  num_outputs = num_outputs
)

config <- list(config_tab = config_tab, config_2D = config_2D)

# Define methods to be applied
methods <- c(
  "Gradient", "GradxInput", "IntegratedGradient_20_zeros", "IntegratedGradient_20_norm",
  "LRP_simple_0", "LRP_epsilon_0.01", "LRP_alpha-beta_1", "LRP_alpha-beta_2",
  "DeepLift_rescale_zeros", "DeepLift_rescale_norm",
  "DeepLift_reveal-cancel_zeros", "DeepLift_reveal-cancel_norm",
  "DeepSHAP"
)

# Generate models
source("utils/utils_validation/preprocess/preprocess.R")
model_config <- preprocess(num_models, num_refs, config, src_dir)

# Run benchmark ----------------------------------------------------------------

source("utils/utils_validation/methods/benchmark.R")
result <- benchmark(methods, model_config, show, src_dir)

# Create plots -----------------------------------------------------------------

cli::cli_h1("Creating Plots")
source("utils/utils_validation/utils.R")

# result with the MAE error
res_error <- result[[1]]

# Set order of methods
res_error$method <- factor(res_error$method, levels = c(
  "Gradient", "GradxInput", "IntegratedGradient",
  "LRP (simple)", "LRP (epsilon)", "LRP (alpha-beta)",
  "DeepLift (rescale)", "DeepLift (reveal-cancel)", "DeepSHAP"
))

# Gradient-based methods (FIGURE 10 (a))
p <- plot_grad_based(res_error)
p
ggsave("figures/FIGURE_10_a.pdf", p, width = 5, height = 5)

# DeepLift (FIGURE 10 (b))
p <- plot_deeplift(res_error)
p
ggsave("figures/FIGURE_10_b.pdf", p, width = 5, height = 5)

# LRP (FIGURE 10 (c))
p <- plot_lrp(res_error)
p
ggsave("figures/FIGURE_10_c.pdf", p, width = 5, height = 5)

# Outlier analysis -------------------------------------------------------------
cli::cli_h1("Outlier Analysis")

# DeepLift
outliers_deeplift <- res_error[res_error$method_grp == "DeepLift" &
                                 !(res_error$method == "DeepSHAP" & res_error$pooling == "max pooling")  &
                                 res_error$error > 1e-6]
cli_text("Number (Percentage) of models with hyperbolic tangent as activation ",
         "among all DeepLift and DeepSHAP (without max pooling) simulations ",
         "exceeding an error of 1e-6: \n",
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

time_diff <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("Total execution time: ", col_blue(time_diff), " mins\n")


################################################################################
#
#               CODE FOR SECTION 5.2 ("Runtime comparison")
#                     Figure 11 (a) & (b)
#               AND ADDITIONAL FIGURES FROM APPENDIX D.3
#                    Figure 14, 15, 16, 17 & 18
#
################################################################################
start_time <- Sys.time()
set.seed(42)

# Load LaTeX font (Latin modern), only relevant for setting the fonts as in the
# paper, but requires the latinmodern-math font
font_add("LModern_math", "additional_files/latinmodern-math.otf")
showtext_auto()

# NOTE:
# For a minimal test execution in significantly less time, the number of models
# per architecture can be reduced, i.e. set `num_models <- 1` for the global
# attributes. In addition, the step size of the varying parameter can also be
# lowered, e.g., for the number of hidden layers, set `c(1, 4, 8, 12, 16)`
# instead of `c(1, seq(2, 20, by = 2))`.

# # Minimal execution
num_models <- 1
num_refs <- c(4)
IMAGE_SHAPES <- lapply(c(16, seq(4) * 64), function(x) c(x,x,3))
NUM_OUTPUTS <- c(1, seq(4, 16, by = 4))
NUM_HIDDEN_LAYERS <- c(1, seq(4, 16, by = 4))
NUM_HIDDEN_UNITS <- list(c(128, seq(512, 2048, by = 512)), c(10, seq(100, 400, by = 100)))
BATCH_SIZES <- list(c(32, seq(128, 512, by = 128)), c(16, seq(64, 256, by = 64)))
show <- FALSE # only for debugging

## Paper execution
# num_models <- 20
# num_refs <- c(10)
# IMAGE_SHAPES <- lapply(c(16, seq(10) * 32), function(x) c(x,x,3))
# NUM_OUTPUTS <- c(1, seq(2, 20, by = 2))
# NUM_HIDDEN_LAYERS <- c(1, seq(2, 20, by = 2))
# NUM_HIDDEN_UNITS <- list(c(128, seq(256, 2560, by = 256)), c(10, seq(50, 500, by = 50)))
# BATCH_SIZES <- list(c(32, seq(64, 640, by = 64)), c(16, seq(32, 320, by = 32)))

# Load utility functions
source("utils/utils_validation/utils_time.R")


# Time comparison for output nodes (FIGURE 11 (b) & 14) ------------------------
cli_h1("Comparison of output nodes")
num_outputs <- NUM_OUTPUTS
num_hidden_layers <- c(2)
num_hidden_units <- list(c(128), c(5))  # first for tabular (units) and second for image (filters)
batch_sizes <- c(16)
src_dir <- "tmp_results/5_2_Num_Outputs"

res <- compare_time(num_models, num_refs, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes, show = show)
#+ fig.width=18, fig.height=16
create_plots(res, "num_outputs")
#+ fig.width=18, fig.height=8
create_figure_plot(res, "num_outputs")


# Time comparison for number layers (FIGURE 11 (a) & 15) -----------------------
cli_h1("Comparison of number of layers")
num_outputs <- c(1)
num_hidden_layers <- NUM_HIDDEN_LAYERS
num_hidden_units <-  list(c(128), c(5))
batch_sizes <- c(16)
src_dir <- "tmp_results/5_2_Num_Layers"

res <- compare_time(num_models, num_refs, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes, show = show)
#+ fig.width=18, fig.height=16
create_plots(res, "hidden_depth")
#+ fig.width=18, fig.height=8
create_figure_plot(res, "hidden_depth")


# Time comparison for hidden width (FIGURE 16) ---------------------------------
cli_h1("Comparison of hidden width")
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- NUM_HIDDEN_UNITS
batch_sizes <- c(16)
src_dir <- "tmp_results/5_2_Num_Units"

res <- compare_time(num_models, num_refs, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes, show = show)
#+ fig.width=18, fig.height=16
create_plots(res, "hidden_width")


# Time comparison for batch_size (FIGURE 17) -----------------------------------
cli_h1("Comparison of batch size")
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- list(c(128), c(5))
batch_sizes <- BATCH_SIZES
src_dir <- "tmp_results/5_2_Batch_Size"

res <- compare_time(num_models, num_refs, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes, show = show)
#+ fig.width=18, fig.height=16
create_plots(res, "batch_size")

# Time comparison for image size (FIGURE 18) -----------------------------------
cli_h1("Comparison of image size")
num_outputs <- c(1)
num_hidden_layers <- c(2)
num_hidden_units <- list(c(128), c(5))
image_shapes <- IMAGE_SHAPES
batch_sizes <- c(16)
src_dir <- "tmp_results/5_2_Image_size"

res <- compare_time(num_models, num_refs, num_outputs, num_hidden_layers,
                    num_hidden_units, src_dir, batch_sizes,
                    image_shapes = image_shapes, show = show)
#+ fig.width=18, fig.height=16
create_plots(res, "num_inputs")

# Print execution time
time_diff <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("\nTotal execution time: ", col_blue(time_diff), " mins\n")


################################################################################
#
#           CODE FOR APPENDIX E ("LRP with bias for innvestigate")
#
################################################################################
start_time <- Sys.time()

# Utility functions ------------------------------------------------------------
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


# Problem: LRP in innvestigate with bias in linear layers ----------------------
input <- layer_input(shape = c(1))
output <- input %>%
  layer_dense(1, use_bias = TRUE, weights = list(array(1, dim = c(1, 1)), array(-0.25)))
model <- keras_model(input, output)

# Check if dir exists
if (!dir.exists("tmp_results/Appendix_E")) {
  dir.create("./tmp_results/Appendix_E", recursive = TRUE)
}

model_path <- "tmp_results/Appendix_E/model_LRP_with_bias.h5"
model$save(model_path)

input <- array(1, dim = c(1, 1))
res_innv <- r(func_innv, args = list(input = input, model_path = model_path))
res_innsight <- func_innsight(input, model_path)

cli_h1("Results")
cli_h3("iNNvestigate")
res_innv
cli_h3("innsight")
res_innsight

time_diff <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("\nTotal execution time: ", col_blue(time_diff), " mins\n")


###############################################################################
#                             Print sessionInfo()
###############################################################################
sessionInfo()



