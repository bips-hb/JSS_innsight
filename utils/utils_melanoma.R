encode_categorical <- function(df, img_source) {
  if (is.null(df$target)) {
    df$target = NA
  }

  data.frame(
    image_name = df$image_name,
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

load_images <- function(image_files) {
  library("reticulate")

  tf <- import("tensorflow")
  np <- import("numpy")

  images <- lapply(image_files,
                   function(image_file) {
                     image_tf = tf$keras$preprocessing$image$load_img(paste0("additional_files/", image_file, ".jpg"))
                     image_tf = tf$keras$preprocessing$image$img_to_array(image_tf)
                     image_tf = tf$image$resize(image_tf, c(224L, 224L))$numpy()
                   })

  images <- np$stack(images) / 255
  images[,,,1] <- images[,,,1] - 0.80612123
  images[,,,2] <- images[,,,2] - 0.62106454
  images[,,,3] <- images[,,,3] - 0.591202

  images
}
