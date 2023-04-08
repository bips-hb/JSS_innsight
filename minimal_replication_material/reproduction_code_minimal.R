#
# NOTE: This is a minimal R script to reproduce the figures from the JSS paper
# "Interpreting Deep Neural Networks with the R Package innsight". Since the
# results shown in the paper for the validity and runtime of the package
# (Section 5) require different Python environments, they cannot be covered in
# this minimal R script. For this, we refer to our GitHub repository at
# https://github.com/bips-hb/JSS_innsight, which explains the reproduction of
# all figures in detail and provides the necessary code. Nevertheless, the
# figures from Section 4 can be reproduced with the following code snippets.
#

###############################################################################
#                         CODE FOR SECTION 4.1
#                     FIGURE 7 (a) AND FIGURE 7 (b)
###############################################################################

# Load necessary libraries
library("palmerpenguins")
library("neuralnet")
library("innsight")
library("ggplot2")

# The torch package must be installed correctly
if (!torch::torch_is_installed()) stop("Call `torch::install_torch()`")

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
conv <- Converter$new(model)

# Apply the LRP method with the alpha-beta-rule (alpha = 2)
# The output variable (species) must be removed from the test_data
lrp <- LRP$new(conv, test_data, rule_name = "alpha_beta", rule_param = 2)

# Plot the relevance scores for two outputs of the neural network model
# This creates the FIGURE 7(a)
plot(lrp, data_idx = c(1, 76), output_idx = c(1, 3)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6))

# Create a box plot of the relevance scores for two outputs of the neural
# network model using the innsight package
# This creates the FIGURE 7 (b)
boxplot(lrp, output_idx = c(1, 3), preprocess_FUN = identity,
        ref_data_idx = 1) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6))


###############################################################################
#                         CODE FOR SECTION 4.2
#                           FIGURE 9 (b)
###############################################################################

# Load necessary packages
library("innsight")
library("ggplot2")
library("keras")

# Make sure TensorFlow is installed
if (!is_keras_available()) stop("Call `keras::install_keras()`")

# Set the URL for the model and download the model
model_url <- "https://github.com/bips-hb/JSS_innsight/raw/master/minimal_replication_material/melanoma_model.h5"
model_path <- get_file("JSS_innsight_melanoma_model.h5",
                       origin = model_url,
                       cache_subdir = "models")

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
converter <- Converter$new(model, input_names = input_names,
                           output_names = output_name)

# Download input data
input_url <- "https://github.com/bips-hb/JSS_innsight/raw/master/minimal_replication_material/melanoma_inputs.rds"
download.file(input_url, "tmp_df.rds")
# Load inputs
input <- readRDS("tmp_df.rds")
# Delete the temporary file
unlink("tmp_df.rds")

# Set the LRP rules and apply the method
rule_name <- list(Conv2D_Layer = "alpha_beta", Dense_Layer = "epsilon")
rule_param <- list(Conv2D_Layer = 1.5, Dense_Layer = 0.01)
res <- LRP$new(converter, input, channels_first = FALSE,
               rule_name = rule_name, rule_param = rule_param)

# Create the S4 'innsight_ggplot2' object (the suggested packages 'gridExtra'
# and 'gtable' are required for advanced plots)
require("gridExtra")
require("gtable")
p <- plot(res, data_idx = c(2, 3, 1)) + theme_bw()

# Modify the plot layout
# Change facet labels in the plot top left
p[1, 1] <- p[1, 1, restyle = FALSE] +
  facet_grid(cols = vars(model_input),
             labeller = as_labeller(c(Input_1 = "Image input")))
# Change the facet labels in the plot top right
p[1, 2] <- p[1, 2, restyle = FALSE] +
  facet_grid(rows = vars(data),
             cols = vars(model_input),
             labeller = as_labeller(c(data_2 = "ISIC_6535558 (87.82%)",
                                      Input_2 = "Tabular input")))
# Change the facet labels in the lower right plots and rotate the x-axis labels
p[2:3, 2] <- p[2:3, 2, restyle = FALSE] +
  facet_grid(rows = vars(data),
             labeller = as_labeller(c(data_3 = "ISIC_7291021 (0.05%)",
                                      data_1 = "ISIC_0946787 (47.47%)"))) +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6))

# Plot the modified plot (all additional arguments, e.g. 'heights' are
# forwarded to the method 'gridExtra::arrangeGrob()')
# This creates FIGURE 9 (b)
plot(p, heights = c(0.31, 0.31, 0.38))

