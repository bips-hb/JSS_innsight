library(palmerpenguins)
library(innsight)
library(neuralnet)
library(cli)

set.seed(42)

#------------------------- Load data and preprocess ----------------------------

# select only numerical columns 'bill_length_mm', 'bill_depth_mm', 'flipper_length' and
# 'body_mass_g' and the species
data <- na.omit(penguins[, c(1,3,4,5,6)])

# Normalize data
data[, 2:5] <- scale(data[, 2:5])

# We use 10% of the data as test data
train_idx <- sample.int(nrow(data), as.integer(nrow(data) * 0.75))
train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

#--------------------- Train a model with neuralnet ----------------------------

model <- neuralnet(species ~ .,
                   data = train_data,
                   hidden = 128,
                   act.fct = "logistic",
                   err.fct = "ce",
                   linear.output = FALSE)

# Print test accuracy
pred <- apply(predict(model, test_data), 1, function(x) which(x == max(x)))
test_acc <- round(mean(pred == as.numeric(test_data$species)) * 100, 2)
cli_text("Test accuracy: {test_acc}%")

#------------------------- Convert model (Step 1) ------------------------------

conv <- Converter$new(model)

#------------------------- Apply methods (Step 2) ------------------------------

# Apply method 'LRP' with epsilon-rule
lrp <- LRP$new(conv, test_data[, -1],
               rule_name = "alpha_beta", rule_para = 2)

#----------------------- Generate figures (Step 3) -----------------------------
library(ggplot2)

p <- plot(lrp, data_idx = c(1, 76), output_idx = c(1,3)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6))
ggsave("7_Illustration/7_1_Palmerpenguins/penguin_plot.pdf", print(p), width = 5, height = 5)

p <- boxplot(lrp, output_idx = c(1,3), preprocess_FUN = identity, ref_data_idx = 1) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6))
ggsave("7_Illustration/7_1_Palmerpenguins/penguin_boxplot.pdf", print(p), width = 5, height = 5)

