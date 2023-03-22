
###############################################################################
#                            Utility methods
###############################################################################

encode_categorical <- function(df, img_source) {
  if (is.null(df$target)) {
    df$target = NA
  }

  data.frame(
    image_name = paste0(img_source, df$image_name, ".npy"),
    sex_male = as.numeric(df$sex == "male"),
    sex_female = as.numeric(df$sex == "female"),
    age = df$age_approx / 90.0 - 0.5430002,
    loc_head_neck = as.numeric(df$anatom_site_general_challenge == "head/neck"),
    loc_torso = as.numeric(df$anatom_site_general_challenge == "torso"),
    loc_upper_extrem = as.numeric(df$anatom_site_general_challenge == "upper extremity"),
    loc_lower_extrem = as.numeric(df$anatom_site_general_challenge == "lower extremity"),
    loc_palms_soles = as.numeric(df$anatom_site_general_challenge == "palms/soles"),
    loc_oral_genital = as.numeric(df$anatom_site_general_challenge == "oral/genital"),
    loc_missing = as.numeric(df$anatom_site_general_challenge == ""),
    target = df$target
  )
}

load_np_arrays <- function(paths) {
  library(reticulate)
  np <- import("numpy", convert=FALSE)

  as.array(np$stack(lapply(paths, function(arr_path) np$load(arr_path))))
}

get_input <- function(df, index) {
  df <- encode_categorical(df[index, ], "4_Illustrations/4_2_Melanoma/data/train/")
  img_data <- load_np_arrays(df$image_name)
  tab_data <- as.matrix(df[, 2:11])
  colnames(tab_data) <- NULL

  list(img_data, tab_data)
}


save_images <- function(df, images) {
  images <- images[[1]]

  for (i in seq_len(dim(images)[1])) {
    png(file=
          paste0("4_Illustrations/4_2_Melanoma/figures/", df[i, ]$image_name, "_",
                 df[i,]$benign_malignant, "_", round(df[i, ]$pred, 4),  ".png"),
        width=480, height=480, res=300)
    image <- images[i,,,, drop = FALSE]
    dim(image) <- c(224,224,3)
    image[,,1] <- image[,,1] + 0.80612123
    image[,,2] <- image[,,2] + 0.62106454
    image[,,3] <- image[,,3] + 0.591202
    par(mar=c(0,0,0,0))
    plot.new()
    rasterImage(image, xleft = -0.04, xright = 1.04,
                ytop = -0.04, ybottom = 1.04, interpolate = FALSE)
    dev.off()
  }
}

save_overlayed_images <- function(df, images, result, alpha = 0.6) {
  images <- images[[1]]

  for (i in seq_len(dim(images)[1])) {
    png(file=paste0("4_Illustrations/4_2_Melanoma/figures/", df[i, ]$image_name,
                    "_", df[i,]$benign_malignant, "_", round(df[i, ]$pred, 4),
                    "_overlay.png"),
        width=480, height=480, res=300)

    # Prepare image
    image <- images[i,,,, drop = FALSE]
    dim(image) <- c(224,224,3)
    image[,,1] <- image[,,1] + 0.80612123
    image[,,2] <- image[,,2] + 0.62106454
    image[,,3] <- image[,,3] + 0.591202

    # Prepare mask
    res <- apply(result$result[[1]][[1]][i,,,], c(1,2), sum)
    res <- res / max(abs(res))
    res_array <- array(0, dim = c(224,224,3))
    res_array[,,1] <- res * (res >= 0)
    res_array[,,3] <- abs(res * (res <= 0))

    # Create and save plot
    par(mar=c(0,0,0,0))
    plot.new()
    rasterImage(res_array * alpha + (1 - alpha) * image, xleft = -0.04,
                xright = 1.04, ytop = -0.04, ybottom = 1.04,
                interpolate = FALSE)
    dev.off()
  }
}



