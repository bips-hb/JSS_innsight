library(innsight)
library(keras)
library(reticulate)
keras::use_condaenv("JSS_paper")
set.seed(42)

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
#                       Calculate predictions
###############################################################################
test_df <- tidyr::drop_na(read.csv(FILE_TEST_CSV))

py_utils <- import_from_path("utils_py", here::here("4_Illustrations/4_2_Melanoma/"))
preds <- c(py_utils$get_predictions(model_path, 256, FILE_TEST_CSV))
test_df$pred <- preds

# get indices for the 5 highest and worst true malign
malign_df <- test_df[test_df$target == 1, ]
best_malign_df <- malign_df[order(malign_df$pred, decreasing = TRUE), ][1:5, ]
# get indices for the 5 highest true benign
benign_df <- test_df[test_df$target == 0, ]
best_benign_df <- benign_df[order(benign_df$pred, decreasing = FALSE), ][1:5, ]
# get indices for 20 random with prob ~0.5
df <- test_df[test_df$pred > 0.4 & test_df$pred < 0.6, ]
df <- rbind(df[df$target == 1, ][1:25, ], df[df$target == 0, ][1:25, ])

test_df <- rbind(best_malign_df, best_benign_df, df)
saveRDS(test_df, "4_Illustrations/4_2_Melanoma/figures/test_df.rds")


###############################################################################
#                           Load test instance
###############################################################################
source("4_Illustrations/4_2_Melanoma/utils.R")

# Get inputs
test_idx <- rownames(test_df)
inputs <- get_input(test_df, test_idx)

###############################################################################
#                               Apply method
###############################################################################

rule_name <- list(
  Conv2D_Layer = "alpha_beta",
  Dense_Layer = "epsilon"
)

rule_param <- list(
  Conv2D_Layer = 1.5,
  Dense_Layer = 0.001
)

res <- LRP$new(converter, inputs, channels_first = FALSE,
               rule_name = rule_name,
               rule_param = rule_param)

###############################################################################
#                               Create plots
###############################################################################
library(ggplot2)

p <- plot(res, data_idx = c(1, 12)) + theme_bw()
ggsave(filename="4_Illustrations/4_2_Melanoma/figures/plot_result_1.pdf", plot=plot(p),
       width = 30, height = 17, units = "cm")
p[1,1] <- p[1,1, restyle = FALSE] +
  facet_grid(cols = vars(model_input),
             labeller = as_labeller(c(Input_1 = "Image input")))
p[1,2] <- p[1,2, restyle = FALSE] +
  facet_grid(rows = vars(data), cols = vars(model_input),
             labeller = as_labeller(c(data_1 = "ISIC_0911264 (99.85%)",
                                      Input_2 = "Tabular input")))
p[2,2] <- p[2,2, restyle = FALSE] +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) +
  facet_grid(rows = vars(data),
             labeller = as_labeller(c(data_12 = "ISIC_0572205 (51.76%)")))

ggsave(filename="4_Illustrations/4_2_Melanoma/figures/plot_result_2.pdf", plot=plot(p),
       width = 30, height = 17, units = "cm")

# Save images
save_images(test_df, inputs)

# Save overlayed images
save_overlayed_images(test_df, inputs, res)
