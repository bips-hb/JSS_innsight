###############################################################################
#                         CODE FOR SECTION 4.2
#                           FIGURE 9 (a) and 9 (b)
###############################################################################
start_time <- Sys.time()

# Disable GPU usage (since it is not necessary for th three images and
# requires a more complicated installation including CUDA)
Sys.setenv(CUDA_VISIBLE_DEVICES = "")

# Load necessary packages
library("innsight")
library("ggplot2")
library("keras")
library("cli")
library("gridExtra")

# Make sure TensorFlow is installed
if (!is_keras_available()) stop("Call `keras::install_keras()`")

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
converter <- Converter$new(model, input_names = input_names,
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

# Show the output (these values are used for the labels in the following!)
df

# Set the LRP rules and apply the method
rule_name <- list(Conv2D_Layer = "alpha_beta", Dense_Layer = "epsilon")
rule_param <- list(Conv2D_Layer = 1.5, Dense_Layer = 0.01)
res <- LRP$new(converter, inputs, channels_first = FALSE,
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
  facet_grid(rows = vars(data),
             cols = vars(model_input),
             labeller = as_labeller(c(data_1 = "ISIC_6535558 (87.82%)",
                                      Input_2 = "Tabular input")))
# Change the facet labels in the lower right plots and rotate the x-axis labels
p[2:3, 2] <- p[2:3, 2, restyle = FALSE] +
  facet_grid(rows = vars(data),
             labeller = as_labeller(c(data_2 = "ISIC_7291021 (0.05%)",
                                      data_3 = "ISIC_0946787 (47.47%)"))) +
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
library("png")
library("grid")

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

###############################################################################
#                             Print sessionInfo()
###############################################################################
cli_h1("Session Info")

sessionInfo()

time_diff <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("\nTotal execution time: ", col_blue(time_diff), " mins\n")

