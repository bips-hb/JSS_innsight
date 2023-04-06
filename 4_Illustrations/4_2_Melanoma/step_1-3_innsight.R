library(innsight)
library(keras)
library(reticulate)

set.seed(42)

Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)

###############################################################################
#                            Global attributes
###############################################################################

FILE_TEST_CSV = "/opt/example-data/siim-isic-melanoma/train.csv"
IMAGE_SHAPE <- c(224, 224, 3)
TAB_NAMES <- c("Sex: Male", "Sex: Female", "Age",
                   "Loc: Head/neck", "Loc: Torso", "Loc: Upper extrm.",
                   "Loc: Lower extrem", "Loc: Palms/soles", "Loc: Oral/genital",
                   "Loc: Missing")

###############################################################################
#                        Create 'Converter' object
###############################################################################

# Load keras model
model_path <- paste0('4_Illustrations/4_2_Melanoma/checkpoints/model_', IMAGE_SHAPE[1], '_', IMAGE_SHAPE[2])
keras_model <- load_model_tf(model_path, compile = FALSE)

input_names <- list(
  list(
    paste0("C", seq_len(IMAGE_SHAPE[3])),
    paste0("H", seq_len(IMAGE_SHAPE[1])),
    paste0("W", seq_len(IMAGE_SHAPE[2]))
    ),
  list(TAB_NAMES)
)
output_name <- c("Probability of malignant lesion")

# Create Converter
converter <- Converter$new(keras_model, input_names = input_names,
                           output_names = output_name)

###############################################################################
#                           Load test instance
###############################################################################
source("4_Illustrations/4_2_Melanoma/utils.R")

# includes the data instances "ISIC_6535558", "ISIC_7291021" and "ISIC_0946787"
test_df <- readRDS("4_Illustrations/4_2_Melanoma/test_df.rds")
# Get inputs
inputs <- get_input(test_df, seq_len(nrow(test_df)))
# Calculate predictions
test_df$pred <- predict(keras_model, inputs)

print(test_df)

###############################################################################
#                               Apply method
###############################################################################

rule_name <- list(
  Conv2D_Layer = "alpha_beta",
  Dense_Layer = "epsilon"
)

rule_param <- list(
  Conv2D_Layer = 1.5,
  Dense_Layer = 0.01
)

res <- LRP$new(converter, inputs, channels_first = FALSE,
               rule_name = rule_name,
               rule_param = rule_param)

###############################################################################
#                               Create plots
###############################################################################
library(ggplot2)

p <- plot(res, data_idx = c(2,3,1)) +
  theme_bw()

p[1, 1] <- p[1, 1, restyle = FALSE] +
  facet_grid(cols = vars(model_input),
             labeller = as_labeller(c(Input_1 = "Image input")))
p[1, 2] <- p[1, 2, restyle = FALSE] +
  facet_grid(rows = vars(data), cols = vars(model_input),
             labeller = as_labeller(c(data_2 = "ISIC_6535558 (87.82%)",
                                      Input_2 = "Tabular input")))
p[2:3, 2] <- p[2:3, 2, restyle = FALSE] +
  facet_grid(rows = vars(data),
             labeller = as_labeller(c(data_3 = "ISIC_7291021 (0.05%)",
                                      data_1 = "ISIC_0946787 (47.47%)"))) +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6))

p <- plot(p, heights = c(0.31, 0.31, 0.38))

ggsave(filename="4_Illustrations/4_2_Melanoma/figures/plot_result_2.pdf", plot=p,
       width = 30, height = 19, units = "cm")

# Save images
save_images(test_df, inputs)
