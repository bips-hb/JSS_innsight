library(innsight)
library(keras)

###############################################################################
#                            Global attributes
###############################################################################

FILE_TEST_CSV = "/home/niklas/Downloads/siim-isic-melanoma/train.csv"#"/opt/example-data/siim-isic-melanoma/test.csv"
IMAGE_SHAPE <- c(224, 224, 3)
TAB_NAMES <- c("sex", "age", "location")
TAB_NAMES_DEC <- c("sex_male", "sex_female", "age",
                   "loc_head_neck", "loc_torso", "loc_upper_extrem",
                   "loc_lower_extrem", "loc_palms_soles", "loc_oral_genital",
                   "loc_missing")

###############################################################################
#                        Create 'Converter' object
###############################################################################

# Load keras model
model_path <- paste0('4_Illustrations/4_2_Melanoma/checkpoints/model_', IMAGE_SHAPE[1], '_', IMAGE_SHAPE[2])
if (dir.exists(model_path)) {
  keras_model <- load_model_tf(model_path)
} else {
  stop("Can not load model from path '", model_path, "'!")
}

input_names <- list(
  list(
    paste0("C", seq_len(IMAGE_SHAPE[3])),
    paste0("H", seq_len(IMAGE_SHAPE[1])),
    paste0("W", seq_len(IMAGE_SHAPE[2]))
    ),
  list(TAB_NAMES_DEC)
)

# Create Converter
converter <- Converter$new(keras_model, input_names = input_names)

###############################################################################
#                           Load test instance
###############################################################################
source("4_Illustrations/4_2_Melanoma/utils.R")

mean(test_df[test_df$target == 1, ]$age_approx)
mean(test_df$age_approx, na.rm = TRUE)

index <- c(2) # 4363 1818 1810 5302

test_df <- read.csv(FILE_TEST_CSV)

input <- get_input(test_df, index)

rule_name <- list(
  Conv2D_Layer = "alpha_beta"
)

rule_param <- list(
  Conv2D_Layer = 2
)

grad <- LRP$new(converter, input, channels_first = FALSE, rule_name = rule_name,
                rule_param = rule_param)

grad <- Gradient$new(converter, input, channels_first = FALSE, times_input = FALSE)

p <- plot(grad, as_plotly = FALSE)
p[1,1] <- p[1,1] + scale_fill_gradient2()

image <- input[[1]]
dim(image) <- c(224,224,3)
image[,,1] <- image[,,1] + 0.80612123
image[,,2] <- image[,,2] + 0.62106454
image[,,3] <- image[,,3] + 0.591202
plot.new()
rasterImage(image, xleft = 0, xright = 1,
            ytop = 0, ybottom = 1, interpolate = FALSE)



