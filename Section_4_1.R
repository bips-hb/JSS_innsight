###############################################################################
#                         CODE FOR SECTION 4.1
#                     FIGURE 7 (a) AND FIGURE 7 (b)
###############################################################################
start_time <- Sys.time()


# Load necessary libraries
library("palmerpenguins")
library("neuralnet")
library("innsight")
library("ggplot2")
library("cli")

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
p <- plot(lrp, data_idx = c(1, 76), output_idx = c(1, 3)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6))

p

# Save figure
ggsave("figures/FIGURE_7_a.pdf", print(p), width = 5, height = 5)

# Create a box plot of the relevance scores for two outputs of the neural
# network model using the innsight package
# This creates the FIGURE 7 (b)
p <- boxplot(lrp, output_idx = c(1, 3), preprocess_FUN = identity,
        ref_data_idx = 1) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6))

p

# Save figure
ggsave("figures/FIGURE_7_b.pdf", print(p), width = 5, height = 5)

###############################################################################
#                             Print sessionInfo()
###############################################################################
cli_h1("Session Info")

sessionInfo()

time_diff <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("\nTotal execution time: ", col_blue(time_diff), " mins\n")

