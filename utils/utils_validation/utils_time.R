
################################################################################
#
#   Utility functions for the time comparison
#
################################################################################

compare_time <- function(num_models, num_refs, num_outputs, num_hidden_layers,
                         num_hidden_units, src_dir, batch_sizes, n_cpu = 1, show = FALSE,
                         image_shapes = NULL) {
  if (!is.list(batch_sizes)) {
    batch_sizes <- list(batch_sizes, batch_sizes)
  }

  # Define experiment config for tabular data
  config_tab = expand.grid(
    input_shape = lapply(batch_sizes[[1]], function(x) c(x, 10)),
    bias = c(TRUE),
    act = c("relu"),
    hidden_depth = num_hidden_layers,
    hidden_width = num_hidden_units[[1]],
    num_outputs = num_outputs
  )

  if (!is.null(image_shapes)) {
    config_tab <- data.frame()
    image_shape <- lapply(image_shapes, function(x) c(batch_sizes[[2]], x))
  } else {
    image_shape <- lapply(batch_sizes[[2]], function(x) c(x, 64, 64, 3))
  }

  # Define experiment config for image data
  config_2D <- expand.grid(
    input_shape = image_shape,
    bias = c(TRUE),
    act = c("relu"),
    batchnorm = "none",
    hidden_depth = num_hidden_layers,
    hidden_width = num_hidden_units[[2]],
    pooling = "none",
    num_outputs = num_outputs
  )

  config <- list(config_tab = config_tab, config_2D = config_2D)

  # Define methods to be applied
  methods <- c(
    "Gradient", "GradxInput", "IntegratedGradient_10_norm",
    "LRP_simple_0", "LRP_epsilon_0.01", "LRP_alpha-beta_2",
    "DeepLift_rescale_norm", "DeepLift_reveal-cancel_norm",
    "ExpectedGradient_10", "DeepSHAP"
  )

  # Generate models
  source("utils/utils_validation/preprocess/preprocess_time.R")
  model_config <- preprocess(num_models, num_refs, config, src_dir)

  # start benchmark
  source("utils/utils_validation/methods/benchmark.R")
  result <- benchmark_time(methods, model_config, show, src_dir, n_cpu)

  # result with the MAE error
  res_time <- get_results_time(src_dir)

  res_time$method_grp[res_time$method_grp %in% c("IntegratedGradient", "ExpectedGradient")] <- "Gradient-based (advanced)"
  res_time$method_grp[res_time$method_grp == "DeepSHAP"] <- "DeepLift"

  res_time
}


create_plots <- function(res, var_name) {

  cowplot::set_null_device("png")

  res$batch_size <- unlist(lapply(res$input_shape, function(a) a[[1]]))
  res$num_inputs <- unlist(lapply(res$input_shape, function(a) a[[2]]))
  res$method <- factor(res$method, levels = c(
    "Gradient", "GradxInput", "IntegratedGradient", "ExpectedGradient",
    "LRP (simple)", "LRP (epsilon)", "LRP (alpha-beta)",
    "DeepLift (rescale)", "DeepLift (reveal-cancel)", "DeepSHAP"
  ))

  x_label <- switch(var_name,
                    num_outputs = "Number of output nodes",
                    hidden_depth = "Number of hidden layers",
                    hidden_width = "Number of hidden units/filters",
                    batch_size = "Number of input instances",
                    num_inputs = "Image height/width")

  file_name <- switch(var_name,
                      num_outputs = "FIGURE_14",
                      hidden_depth = "FIGURE_15",
                      hidden_width = "FIGURE_16",
                      batch_size = "FIGURE_17",
                      num_inputs = "FIGURE_18")

  plots <- list()
  for (time_type in c("time_eval", "time_total")) {
    df <- res[, .(mean = mean(get(time_type))),
              by = c("num_inputs", "pkg", "method", "method_grp", "num_outputs",
                     "hidden_depth", "hidden_width", "batch_size", "data_type")]
    df$data_type <- factor(
      ifelse(df$data_type == "tabular", "Tabular input", "Image input"),
      levels = c("Tabular input", "Image input"))
    df_tab <- df[data_type == "Tabular input"]
    df_img <- df[data_type == "Image input"]
    limits_tab <- if (nrow(df_tab) == 0) c(0,1) else c(min(df_tab$mean), max(df_tab$mean))
    limits_img <- c(min(df_img$mean), max(df_img$mean))
    y_label <- switch(time_type,
                      time_eval = "Evaluation time (sec.)",
                      time_convert = "Conversion time (sec.)",
                      time_total = "Total time (sec.)")

    if (time_type != "time_total") {
      x_lab <- NULL
    } else {
      x_lab <- x_label
    }

    # Gradient-based
    p1_1 <- create_basic_plot(df_tab[method_grp == "Gradient-based", ],
                              "Gradient", var_name, limits_tab, NULL, y_label) +
      theme(plot.margin = unit(c(5.5,1,3,5.5), "pt"),
            strip.background.y = element_blank(),
            strip.text.y = element_blank())
    p1_2 <- create_basic_plot(df_img[method_grp == "Gradient-based", ],
                              "Gradient", var_name, limits_img, x_lab, y_label) +
      theme(plot.margin = unit(c(0,1,5.5,5.5), "pt"),
            strip.background.y = element_blank(),
            strip.text.y = element_blank())


    # LRP
    p2_1 <- create_basic_plot(df_tab[method_grp == "LRP", ],
                            "LRP", var_name, limits_tab, NULL, NULL) +
      theme(plot.margin =  unit(c(5.5,1,3,1), "pt"),
            strip.background.y = element_blank(),
            strip.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank())
    p2_2 <- create_basic_plot(df_img[method_grp == "LRP", ],
                              "LRP", var_name, limits_img, x_lab, NULL) +
      theme(plot.margin =  unit(c(0,1,5.5,1), "pt"),
            strip.background.y = element_blank(),
            strip.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank())


    # DeepLift
    p3_1 <- create_basic_plot(df_tab[method_grp == "DeepLift", ],
                            "DeepLift", var_name, limits_tab, NULL, NULL) +
      theme(plot.margin =  unit(c(5.5,1,3,1), "pt"),
            strip.background.y = element_blank(),
            strip.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank())
    p3_2 <- create_basic_plot(df_img[method_grp == "DeepLift", ],
                              "DeepLift", var_name, limits_img, x_lab, NULL) +
      theme(plot.margin =  unit(c(0,1,5.5,1), "pt"),
            strip.background.y = element_blank(),
            strip.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank())

    # Gradient-based (advanced)
    p4_1 <- create_basic_plot(df_tab[method_grp == "Gradient-based (advanced)", ],
                            "Gradient-based (advanced)", var_name, limits_tab, NULL, NULL) +
      theme(plot.margin =  unit(c(5.5,5.5,3,1), "pt"),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank())
    p4_2 <- create_basic_plot(df_img[method_grp == "Gradient-based (advanced)", ],
                              "Gradient-based (advanced)", var_name, limits_img, x_lab, NULL) +
      theme(plot.margin =  unit(c(0,5.5,5.5,1), "pt"),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank())

    if (time_type != "time_eval") {
      p1_1 <- p1_1 + guides(shape = "none", linetype = "none")
      p2_1 <- p2_1 + guides(shape = "none", linetype = "none")
      p3_1 <- p3_1 + guides(shape = "none", linetype = "none")
      p4_1 <- p4_1 + guides(shape = "none", linetype = "none")
    }
    if (var_name != "hidden_width") {
      p1_1 <- p1_1 +
        theme(axis.ticks.x = element_blank(),
              axis.text.x = element_blank())
      p2_1 <- p2_1 +
        theme(axis.ticks.x = element_blank(),
              axis.text.x = element_blank())
      p3_1 <- p3_1 +
        theme(axis.ticks.x = element_blank(),
              axis.text.x = element_blank())
      p4_1 <- p4_1 +
        theme(axis.ticks.x = element_blank(),
              axis.text.x = element_blank())
    }
    if ((var_name != "num_inputs") | (time_type != "time_eval")) {
      p1_2 <- p1_2 +
        guides(shape = "none", linetype = "none") +
        theme(strip.background.x = element_blank(),
              strip.text.x = element_blank())
      p2_2 <- p2_2 +
        guides(shape = "none", linetype = "none") +
        theme(strip.background.x = element_blank(),
              strip.text.x = element_blank())
      p3_2 <- p3_2 +
        guides(shape = "none", linetype = "none") +
        theme(strip.background.x = element_blank(),
              strip.text.x = element_blank())
      p4_2 <- p4_2 +
        guides(shape = "none", linetype = "none") +
        theme(strip.background.x = element_blank(),
              strip.text.x = element_blank())
    }


    if (time_type == "time_total") {
      legend <- get_legend(
        ggplot(df, aes(x = .data[[var_name]], y = mean, colour = pkg)) +
          geom_line() + geom_point() + theme_bw() +
        scale_colour_manual(values = pal_npg(c("nrc"), 1)(6),
                            breaks = c("captum", "deeplift", "innsight",
                                       "innvestigate", "shap", "zennit")) +
          guides(color = guide_legend(nrow = 1, title = "Package"), linetype = "none", shape = "none") +
          theme(legend.position = "top",
                legend.text = element_text(size = 16,
                                           family = "LModern_math"),
                legend.title = element_text(size = 18,
                                            face = "bold", family = "LModern_math"))
      )

      rel_heights <- c(1.01, 1)
    } else {
      rel_heights <- c(1.1, 1)
    }

    if (var_name == "num_inputs") {
      p <- plot_grid(p1_2, p2_2, p3_2, p4_2, ncol = 4,
                     rel_widths = c(1.1,0.95,0.95,1))
    } else {
      p <- plot_grid(p1_1,p2_1,p3_1, p4_1, p1_2, p2_2, p3_2, p4_2, ncol = 4,
                     rel_widths = c(1.1,0.95,0.95,1), rel_heights = rel_heights)
    }


    plots <- append(plots, list(p))
  }

  p <- plot_grid(legend, plots[[1]], plots[[2]], ncol = 1,
                 rel_heights = c(0.07,1.05,0.95),
                 labels = c("", "a)", "b)"))

  plot(p)
  ggsave(paste0("figures/", file_name, ".pdf"), p,
         width = 17, height = 17)
}

create_basic_plot <- function(df, method_type, var_name, limits, xlabel = NULL,
                              ylabel = NULL) {
  pkgs <- sort(unique(df$pkg))
  p <- ggplot(df, aes(x = .data[[var_name]], colour = pkg)) +
    geom_line(aes(y = mean, linetype = method)) +
    geom_point(aes(y = mean, shape = method), size = 2) +
    facet_grid(cols = vars(method_grp), rows = vars(data_type)) +
    theme_bw() +
    scale_colour(pkgs) +
    labs(colour = "Package", x = xlabel, y = ylabel, linetype = NULL, shape = NULL) +
    scale_y_log10(limits = limits, labels = function(x) format(x, scientific = TRUE)) +
    guides(colour = "none") + #, linetype = guide_legend(nrow = 2), shape = guide_legend(nrow = 2)) +
    theme(legend.position="top",
          legend.spacing.x = unit(2, 'pt'),
          legend.margin=margin(b = 0, t = 0, unit='pt'),
          text = element_text(family = "LModern_math",
                              size = 15))

  if (method_type == "Gradient") {
    values <- c(1, 2)
    labels <- c("Gradient", "Gradient\u00D7Input")
  } else if (method_type == "DeepLift") {
    values <- c(1, 2, 4)
    labels <- c("Rescale", "RevealCancel", "DeepSHAP")
  } else if (method_type == "Gradient-based (advanced)") {
    values <- c(1, 2)
    labels <- c("IntegratedGradient", "ExpectedGradient")
  } else {
    values <- c(1,2,4)
    labels <- c("simple rule", "ε-rule", "α-β-rule")
  }

  p +
    scale_shape_manual(values = values, labels = labels) +
    scale_linetype_manual(values = values, labels = labels)
}

create_figure_plot <- function(res, var_name) {
  cowplot::set_null_device("png")

  x_label <- switch(var_name,
                    num_outputs = "Number of output nodes",
                    hidden_depth = "Number of hidden layers")

  file_name <- switch(var_name,
                      hidden_depth = "FIGURE_11_a",
                      num_outputs = "FIGURE_11_b")

  res$batch_size <- unlist(lapply(res$input_shape, function(a) a[[1]]))
  res$num_inputs <- unlist(lapply(res$input_shape, function(a) a[[2]]))
  res$method <- factor(res$method, levels =  c(
    "Gradient", "GradxInput", "IntegratedGradient", "ExpectedGradient",
    "LRP (simple)", "LRP (epsilon)", "LRP (alpha-beta)",
    "DeepLift (rescale)", "DeepLift (reveal-cancel)", "DeepSHAP"
  ))
  res$data_type <- factor(
    ifelse(res$data_type == "tabular", "Tabular input", "Image input"),
    levels = c("Tabular input", "Image input"))
  facet_rows <- vars(data_type)
  rel_widths = c(1.15,1,1,1.05)

  if (var_name == "num_outputs") {
    res <- res[data_type == "Image input", ]
    by_vals =  c("num_inputs", "pkg", "method", "method_grp", "num_outputs",
                 "hidden_depth", "batch_size", "data_type")

  } else {
    by_vals =  c("num_inputs", "pkg", "method", "method_grp", "num_outputs",
                 "hidden_depth", "batch_size", "data_type")
    res <- res[data_type == "Tabular input"]
  }

  df <- res[, .(mean = mean(time_eval)), by = by_vals]
  limits <- c(min(df$mean), max(df$mean))

  pkgs <- sort(unique(df[method_grp == "Gradient-based",]$pkg))
  p1 <- ggplot(df[method_grp == "Gradient-based", ], aes(x = .data[[var_name]], colour = pkg)) +
    geom_line(aes(y = mean, linetype = method)) +
    geom_point(aes(y = mean, shape = method), size = 3) +
    facet_grid(cols = vars(method_grp), rows = facet_rows) +
    theme_bw() +
    scale_colour(pkgs) +
    scale_shape_manual(values = c(1,2), labels = c("Gradient", "Gradient\u00D7Input")) +
    scale_linetype_manual(values = c(1,2), labels = c("Gradient", "Gradient\u00D7Input")) +
    labs(colour = "Package", x = x_label,
         y = "Evaluation time (sec.)", linetype = NULL, shape = NULL) +
    scale_y_log10(limits = limits) +
    guides(colour = "none", fill = "none") +
    theme(legend.position="top",
          legend.spacing.x = unit(2, 'pt'),
          strip.background.y = element_blank(),
          strip.text.y = element_blank(),
          plot.margin =  unit(c(5.5,1,5.5,3), "pt"),
          legend.margin=margin(b = 0, t = 0, unit='pt'),
          text = element_text(family = "LModern_math", size = 14))

  pkgs <- sort(unique(df[method_grp == "DeepLift",]$pkg))
  p3 <- ggplot(df[method_grp == "DeepLift", ], aes(x = .data[[var_name]], colour = pkg)) +
    geom_line(aes(y = mean, linetype = method)) +
    geom_point(aes(y = mean, shape = method), size = 3) +
    facet_grid(cols = vars(method_grp), rows = facet_rows) +
    theme_bw() +
    scale_colour(pkgs) +
    scale_shape_manual(values = c(1,2,4), labels = c("Rescale", "RevealCancel", "DeepSHAP")) +
    scale_linetype_manual(values = c(1,2,4), labels = c("Rescale", "RevealCancel", "DeepSHAP")) +
    labs(colour = "Package", x = x_label,
         y = NULL, linetype = NULL, shape = NULL) +
    scale_y_log10(limits = limits) +
    guides(colour = "none", fill = "none") +
    theme(legend.position="top",
          plot.margin =  unit(c(5.5,1,5.5,1), "pt"),
          legend.spacing.x = unit(2, 'pt'),
          legend.margin=margin(b = 0, t = 0, unit='pt'),
          axis.ticks.y = element_blank(),
          axis.text.y = element_blank(),
          strip.background.y = element_blank(),
          strip.text.y = element_blank(),
          text = element_text(family = "LModern_math", size = 14))

  pkgs <- sort(unique(df[method_grp == "LRP",]$pkg))
  p2 <- ggplot(df[method_grp == "LRP", ], aes(x = .data[[var_name]], colour = pkg)) +
    geom_line(aes(y = mean, linetype = method)) +
    geom_point(aes(y = mean, shape = method), size = 3) +
    facet_grid(cols = vars(method_grp), rows = facet_rows) +
    theme_bw() +
    scale_colour(pkgs) +
    scale_shape_manual(values = c(1,2,4), labels = c("simple rule", "ε-rule", "α-β-rule")) +
    scale_linetype_manual(values = c(1,2,4), labels = c("simple rule", "ε-rule", "α-β-rule")) +
    labs(colour = "Package", x = x_label,
         y = NULL, linetype = NULL, shape = NULL) +
    scale_y_log10(limits = limits) +
    guides(colour = "none", fill = "none") +
    theme(legend.position="top",
          legend.spacing.x = unit(2, 'pt'),
          plot.margin =  unit(c(5.5,1,5.5,1), "pt"),
          legend.margin=margin(b = 0, t = 0, unit='pt'),
          axis.ticks.y = element_blank(),
          axis.text.y = element_blank(),
          strip.background.y = element_blank(),
          strip.text.y = element_blank(),
          text = element_text(family = "LModern_math", size = 14))

  pkgs <- sort(unique(df[method_grp == "Gradient-based (advanced)", ]$pkg))
  p4 <- ggplot(df[method_grp == "Gradient-based (advanced)", ], aes(x = .data[[var_name]], colour = pkg)) +
    geom_line(aes(y = mean, linetype = method)) +
    geom_point(aes(y = mean, shape = method), size = 3) +
    facet_grid(cols = vars(method_grp), rows = facet_rows) +
    theme_bw() +
    scale_colour(pkgs) +
    scale_shape_manual(values = c(1,2), labels = c("IntegratedGradient", "ExpectedGradient")) +
    scale_linetype_manual(values = c(1,2), labels = c("IntegratedGradient", "ExpectedGradient")) +
    labs(colour = "Package", x = x_label,
         y = NULL, linetype = NULL, shape = NULL) +
    scale_y_log10(limits = limits) +
    guides(colour = "none", fill = "none") +
    theme(legend.position="top",
          plot.margin =  unit(c(5.5,1,5.5,1), "pt"),
          legend.spacing.x = unit(2, 'pt'),
          legend.margin=margin(b = 0, t = 0, unit='pt'),
          axis.ticks.y = element_blank(),
          axis.text.y = element_blank(),
          text = element_text(family = "LModern_math", size = 14))

  legend <- get_legend(
    ggplot(df, aes(x = .data[[var_name]], y = mean, colour = pkg)) +
      geom_line() + geom_point() + theme_bw() +
      scale_colour_manual(values = pal_npg(c("nrc"), 1)(6),
                          breaks = c("captum", "deeplift", "innsight",
                                     "innvestigate", "shap", "zennit")) +
      guides(color = guide_legend(nrow = 1, title = "Package"), linetype = "none", shape = "none") +
      theme(legend.position = "top",
            legend.text = element_text(size = 16,
                                       family = "LModern_math"),
            legend.title = element_text(size = 18,
                                        face = "bold", family = "LModern_math"))
  )

  p <- plot_grid(p1,p2,p3, p4, ncol = 4, rel_widths = rel_widths)

  p <- plot_grid(legend, p, ncol = 1, rel_heights = c(0.075,1))

  plot(p)
  ggsave(paste0("figures/", file_name, ".pdf"), p,
         width = 14.5, height = 5.5)
}

scale_fill <- function(pkgs) {
  all_pkgs <- c("captum", "deeplift", "innsight", "innvestigate", "shap", "zennit")
  all_colors <- pal_npg(c("nrc"), 1)(6)

  scale_fill_manual(values = all_colors[which(all_pkgs %in% pkgs)])
}

scale_colour <- function(pkgs) {
  all_pkgs <- c("captum", "deeplift", "innsight", "innvestigate", "shap", "zennit")
  all_colors <- pal_npg(c("nrc"), 1)(6)
  scale_colour_manual(values = all_colors[which(all_pkgs %in% pkgs)])
}
