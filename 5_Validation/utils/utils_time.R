
################################################################################
#
#   Utility functions for the time comparison
#
################################################################################

compare_time <- function(num_models, num_outputs, num_hidden_layers,
                         num_hidden_units, src_dir, batch_sizes, show = TRUE) {
  # Define experiment config for tabular data
  config_tab = expand.grid(
    input_shape = lapply(batch_sizes, function(x) c(x, 10)),
    bias = c(TRUE),
    act = c("relu"),
    hidden_depth = num_hidden_layers,
    hidden_width = num_hidden_units[[1]],
    num_outputs = num_outputs
  )

  # Define experiment config for image data
  config_2D <- expand.grid(
    input_shape = lapply(batch_sizes, function(x) c(x, 10, 10, 3)),
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

  cowplot::set_null_device("png")

  res$batch_size <- unlist(lapply(res$input_shape, function(a) a[[1]]))
  res$num_inputs <- unlist(lapply(res$input_shape, function(a) a[[2]]))

  if (!dir.exists(paste0(src_dir, "/figures"))) {
    dir.create(paste0(src_dir, "/figures"))
  }

  x_label <- switch(var_name,
                    num_outputs = "Number of output nodes",
                    hidden_depth = "Number of hidden layers",
                    hidden_width = "Number of hidden units/filters",
                    batch_size = "Number of input instances")

  for (time_type in c("time_eval", "time_convert", "time_total")) {
    df <- res[, .(mean = mean(get(time_type))),
              by = c("num_inputs", "pkg", "method", "method_grp", "num_outputs",
                     "hidden_depth", "hidden_width", "batch_size", "data_type")]
    df$data_type <- factor(
      ifelse(df$data_type == "tabular", "Tabular input", "Image input"),
      levels = c("Tabular input", "Image input"))
    limits <- c(min(df$mean), max(df$mean))
    y_label <- switch(time_type,
                      time_eval = "Evaluation time (sec.)",
                      time_convert = "Conversion time (sec.)",
                      time_total = "Total time (sec.)")

    # Gradient-based
    if (var_name == "hidden_width") {
      p1 <- create_basic_plot(df[df$method_grp == "Gradient-based" & df$data_type == "Tabular input", ],
                                "Gradient", var_name, limits, NULL, y_label) +
        theme(plot.margin = unit(c(5.5,0,3,5.5), "pt"),
              strip.background.y = element_blank(),
              strip.text.y = element_blank())
      p1_2 <- create_basic_plot(df[df$method_grp == "Gradient-based" & df$data_type == "Image input", ],
                                "Gradient", var_name, limits, x_label, y_label) +
        theme(plot.margin = unit(c(0,0,5.5,5.5), "pt"),
              strip.background = element_blank(),
              strip.text = element_blank()) +
        guides(shape = "none", linetype = "none")
    } else {
      p1 <- create_basic_plot(df[df$method_grp == "Gradient-based", ],
                              "Gradient", var_name, limits, x_label, y_label) +
        theme(plot.margin = unit(c(5.5,0,5.5,5.5), "pt"),
              strip.background.y = element_blank(),
              strip.text.y = element_blank())
    }


    # LRP
    if (var_name == "hidden_width") {
      p2 <- create_basic_plot(df[df$method_grp == "LRP" & df$data_type == "Tabular input", ],
                              "LRP", var_name, limits, NULL, NULL) +
        theme(plot.margin =  unit(c(5.5,0,3,0), "pt"),
              strip.background.y = element_blank(),
              strip.text.y = element_blank(),
              axis.ticks.y = element_blank(),
              axis.text.y = element_blank())
      p2_2 <- create_basic_plot(df[df$method_grp == "LRP" & df$data_type == "Image input", ],
                                "LRP", var_name, limits, x_label, NULL) +
        theme(plot.margin =  unit(c(0,0,5.5,0), "pt"),
              strip.background = element_blank(),
              strip.text = element_blank(),
              axis.ticks.y = element_blank(),
              axis.text.y = element_blank()) +
        guides(shape = "none", linetype = "none")
    } else {
      p2 <- create_basic_plot(df[df$method_grp == "LRP", ],
                              "LRP", var_name, limits, x_label, NULL) +
        theme(plot.margin =  unit(c(5.5,0,5.5,0), "pt"),
              strip.background.y = element_blank(),
              strip.text.y = element_blank(),
              axis.ticks.y = element_blank(),
              axis.text.y = element_blank())
    }

    # DeepLift
    if (var_name == "hidden_width") {
      p3 <- create_basic_plot(df[df$method_grp == "DeepLift" & df$data_type == "Tabular input", ],
                              "DeepLift", var_name, limits, NULL, NULL) +
        theme(plot.margin =  unit(c(5.5,5.5,3,0), "pt"),
              axis.ticks.y = element_blank(),
              axis.text.y = element_blank())
      p3_2 <- create_basic_plot(df[df$method_grp == "DeepLift" & df$data_type == "Image input", ],
                                "DeepLift", var_name, limits, x_label, NULL) +
        theme(plot.margin =  unit(c(0,5.5,5.5,0), "pt"),
              strip.background.x = element_blank(),
              strip.text.x = element_blank(),
              axis.ticks.y = element_blank(),
              axis.text.y = element_blank()) +
        guides(shape = "none", linetype = "none")
    } else {
      p3 <- create_basic_plot(df[df$method_grp == "DeepLift", ],
                              "DeepLift", var_name, limits, x_label, NULL) +
        theme(plot.margin =  unit(c(5.5,5.5,5.5,0), "pt"),
              axis.ticks.y = element_blank(),
              axis.text.y = element_blank())
    }


    legend <- get_legend(
      p1 +
        guides(color = guide_legend(nrow = 1), linetype = "none", shape = "none") +
        theme(legend.position = "top",
              legend.text = element_text(size=18, family = "LModern_math" ),
              legend.title = element_text(size=18, face = "bold", family = "LModern_math"))
    )

    if (var_name == "hidden_width") {
      p <- plot_grid(p1,p2,p3, p1_2, p2_2, p3_2, ncol = 3,
                rel_widths = c(1.09,0.91,1), rel_heights = c(1.05, 0.95))
    } else {
      p <- plot_grid(p1,p2,p3, ncol = 3, rel_widths = c(1.09,0.91,1))
    }

    p <- plot_grid(legend, p, ncol = 1, rel_heights = c(0.05,1))

    ggsave(paste0(src_dir, "/figures/time_", var_name, "_", time_type, ".pdf"), p,
           width = 14, height = 8)
  }
}

create_basic_plot <- function(df, method_type, var_name, limits, xlabel = NULL, ylabel = NULL) {
  pkgs <- sort(unique(df$pkg))
  p <- ggplot(df, aes(x = .data[[var_name]], colour = pkg)) +
    geom_line(aes(y = mean, linetype = method)) +
    geom_point(aes(y = mean, shape = method), size = 3) +
    facet_grid(cols = vars(method_grp), rows = vars(data_type)) +
    theme_bw() +
    scale_colour(pkgs) +
    labs(colour = "Package", x = xlabel, y = ylabel, linetype = NULL, shape = NULL) +
    scale_y_log10(limits = limits) +
    guides(colour = "none") +
    theme(legend.position="top",
          legend.margin=margin(b = 0, t = 0, unit='pt'),
          text = element_text(family = "LModern_math", size = 15))

  if (method_type == "Gradient") {
    values <- c(1, 2)
    labels <- c("Gradient", "Gradient\u00D7Input")
  } else if (method_type == "DeepLift") {
    values <- c(1, 2)
    labels <- c("Rescale", "RevealCancel")
  } else {
    values <- c(1,2,4)
    labels <- c("simple rule", expression(epsilon*"-rule"),
                expression(alpha*"-"*beta*"-rule"))
  }

  p +
    scale_shape_manual(values = values, labels = labels) +
    scale_linetype_manual(values = values, labels = labels)
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
