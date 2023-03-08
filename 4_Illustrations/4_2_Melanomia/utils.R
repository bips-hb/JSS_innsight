
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
    age = df$age_approx,
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
  df <- encode_categorical(df[index, ], "4_Illustrations/data/test/")
  img_data <- load_np_arrays(df$image_name)
  tab_data <- as.matrix(df[, 2:11])
  colnames(tab_data) <- NULL

  list(img_data, tab_data)
}
