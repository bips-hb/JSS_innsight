
################################################################################
#
#   Utility functions for the time comparison
#
################################################################################

compare_time <- function(num_models, num_outputs, num_inputs, num_hidden_layers,
                         num_hidden_units, src_dir, batch_sizes, show = TRUE) {
  input_shapes <- expand.grid(batch_sizes, num_inputs)
  input_shapes <- lapply(seq_len(nrow(input_shapes)),
                         function(i) c(input_shapes[i,1], input_shapes[i, 2]))

  # Define experiment configs
  config_tab = expand.grid(
    input_shape = input_shapes,
    bias = c(TRUE),
    act = c("relu"),
    hidden_depth = num_hidden_layers,
    hidden_width = num_hidden_units,
    num_outputs = num_outputs
  )

  config <- list(config_tab = config_tab, config_2D = data.frame())

  # Define methods to be applied
  methods <- c(
    "Gradient", "GradxInput",
    "LRP_simple_0", "LRP_epsilon_0.01", "LRP_alpha-beta_2",
    "DeepLift_rescale_norm",
    "DeepLift_reveal-cancel_norm"
  )

  # Generate models
  source("5_Validation/utils/preprocess/preprocess_time.R")
  model_config <- preprocess(num_models, config, src_dir)

  # start benchmark
  source("5_Validation/utils/methods/benchmark.R")
  result <- benchmark(methods, model_config, show, src_dir)

  # result with the MAE error
  res_time <- get_results_time(src_dir)

  res_time
}


create_plots <- function(res, var_name, src_dir) {
  library(ggsci)
  library(cowplot)

  res$batch_size <- unlist(lapply(res$input_shape, function(a) a[[1]]))
  res$num_inputs <- unlist(lapply(res$input_shape, function(a) a[[2]]))

  df <- res[, .(mean = mean(time_eval),
                lower = quantile(time_eval, probs = 0.1),
                upper = quantile(time_eval, probs = 0.9)),
            by = c("num_inputs", "pkg", "method", "method_grp", "num_outputs",
                   "hidden_depth", "hidden_width", "batch_size")]

  if (!dir.exists(paste0(src_dir, "/figures"))) {
    dir.create(paste0(src_dir, "/figures"))
  }

  x_label <- switch(var_name,
                    num_outputs = "Number of output nodes",
                    hidden_depth = "Number of hidden layers",
                    hidden_width = "Number of hidden units",
                    batch_size = "Number of input instances")

  limits <- c(min(df$mean), max(df$mean))

  # Gradient-based
  pkgs <- sort(unique(df[df$method_grp == "Gradient-based", ]$pkg))
  p1 <- ggplot(df[df$method_grp == "Gradient-based", ], aes(x = .data[[var_name]], colour = pkg)) +
    geom_line(aes(y = mean, linetype = method)) +
    geom_point(aes(y = mean, shape = method), size = 3) +
    facet_grid(cols = vars(method_grp)) +
    theme_bw() +
    theme(legend.position="top", plot.margin = unit(c(5.5,0,5.5,5.5), "pt"),
          legend.margin=margin(b = 0, t = 0, unit='pt')) +
    scale_colour(pkgs) +
    ylab("Evaluation time (sec.)") +
    labs(colour = "Package", x = x_label, linetype = NULL, shape = NULL) +
    scale_y_log10(limits = limits) +
    guides(colour = "none")

  # LRP
  pkgs <- sort(unique(df[df$method_grp == "LRP", ]$pkg))
  p2 <- ggplot(df[df$method_grp == "LRP", ], aes(x = .data[[var_name]], color = pkg)) +
    geom_line(aes(y = mean, linetype = method)) +
    geom_point(aes(y = mean, shape = method), size = 3) +
    facet_grid(cols = vars(method_grp)) +
    theme_bw() +
    theme(legend.position="top", axis.ticks.y = element_blank(),
          plot.margin = unit(c(5.5,0,5.5,0), "pt"),
          legend.margin=margin(b = 0, t = 0, unit='pt')) +
    scale_colour(pkgs) +
    ylab(NULL) +
    labs(colour = "Package", x = x_label, linetype = NULL, shape = NULL) +
    scale_y_log10(limits = limits, labels = NULL) +
    guides(colour = "none")

  # DeepLift
  pkgs <- sort(unique(df[df$method_grp == "DeepLift", ]$pkg))
  p3 <- ggplot(df[df$method_grp == "DeepLift", ], aes(x = .data[[var_name]], color = pkg)) +
    geom_line(aes(y = mean, linetype = method)) +
    geom_point(aes(y = mean, shape = method), size = 3) +
    facet_grid(cols = vars(method_grp)) +
    theme_bw() +
    theme(legend.position="top", axis.ticks.y = element_blank(),
          plot.margin = unit(c(5.5,5.5,5.5,0), "pt"),
          legend.margin=margin(b = 0, t = 0, unit='pt')) +
    scale_colour(pkgs) +
    ylab(NULL) +
    labs(colour = "Package", x = x_label, linetype = NULL, shape = NULL) +
    scale_y_log10(limits = limits, labels = NULL) +
    guides(colour = "none")

  legend <- get_legend(
    p1 +
      guides(color = guide_legend(nrow = 1), linetype = "none", shape = "none") +
      theme(legend.position = "top",
            legend.text = element_text(size=15),
            legend.title = element_text(size=15, face = "bold"))
  )

  p <- plot_grid(p1,p2,p3, ncol = 3)
  p <- plot_grid(legend, p, ncol = 1, rel_heights = c(0.05,1))

  ggsave(paste0(src_dir, "/figures/time_", var_name, ".pdf"), p,
         width = 14, height = 8)
}

scale_fill <- function(pkgs) {
  all_pkgs <- c("captum", "deeplift", "innsight", "innvestigate", "zennit")
  all_colors <- pal_npg(c("nrc"), 1)(5)

  scale_fill_manual(values = all_colors[which(all_pkgs %in% pkgs)])
}

scale_colour <- function(pkgs) {
  all_pkgs <- c("captum", "deeplift", "innsight", "innvestigate", "zennit")
  all_colors <- pal_npg(c("nrc"), 1)(5)
  scale_colour_manual(values = all_colors[which(all_pkgs %in% pkgs)])
}
