################################################################################
#                           Generate plots
################################################################################

plot_grad_based <- function(res_error) {
  ggplot(res_error[res_error$method_grp %in% c("Gradient-based"), ]) +
    geom_boxplot(aes(y = error, x = pkg, fill = method),
                 outlier.size = 0.75,
                 outlier.alpha = 0.25) +
    facet_grid(cols = vars(method_grp)) +
    scale_y_continuous(trans = log10_with_0_trans(9), limits = c(0, 1e0)) +
    add_gray_box() +
    scale_fill_manual(values = pal_npg(c("nrc"), 1)(9)[1:3],
                      labels = c("Gradient", "Gradient\u00D7Input", "IntegratedGradient")) +
    geom_hline(yintercept = 0, alpha = 0.5) +
    labs(y = "Mean absolute difference",
         x = "Package",
         fill = NULL) +
    theme_bw() +
    theme(
      legend.position = "top",
      legend.margin = margin(),
      legend.spacing.x = unit(4, 'pt'),
      text = element_text(family = "LModern_math", size = 14))
}

plot_deeplift <- function(res_error) {
  ggplot(res_error[res_error$method_grp %in% c("DeepLift"), ]) +
    geom_boxplot(aes(y = error, x = pkg, fill = method),
                 outlier.size = 0.75,
                 outlier.alpha = 0.25) +
    facet_grid(cols = vars(method_grp), rows = vars(pooling)) +
    scale_y_continuous(trans = log10_with_0_trans(9), limits = c(0, 1e0)) +
    add_gray_box() +
    scale_fill_manual(values = pal_npg(c("nrc"), 1)(9)[4:6],
                      labels = c("Rescale", "RevealCancel", "DeepSHAP")) +
    geom_hline(yintercept = 0, alpha = 0.5) +
    labs(y = NULL, x = "Package", fill = NULL) +
    theme_bw() +
    theme(legend.position="top",
          legend.margin = margin(),
          legend.spacing.x = unit(4, 'pt'),
          text = element_text(family="LModern_math", size = 14))
}

plot_lrp <- function(res_error) {
  ggplot(res_error[res_error$method_grp %in% c("LRP"), ]) +
    geom_boxplot(aes(y = error, x = pkg, fill = method),
                 outlier.size = 0.75,
                 outlier.alpha = 0.25) +
    facet_grid(cols = vars(method_grp), rows = vars(bias)) +
    scale_y_continuous(trans = log10_with_0_trans(9, 4), limits = c(0, 1e2)) +
    add_gray_box() +
    geom_hline(yintercept = 0, alpha = 0.5) +
    labs(y = NULL, x = "Package", fill = NULL) +
    theme_bw() +
    theme(legend.position="top",
          legend.margin = margin(),
          legend.spacing.x = unit(4, 'pt'),
          text = element_text(family="LModern_math", size = 15))+
    scale_fill_manual(values = pal_npg(c("nrc"), 1)(9)[7:9],
                      labels = c("simple rule", expression(epsilon*"-rule"),
                                 expression(alpha*"-"*beta*"-rule")))
}

################################################################################
#                         Utility functions
################################################################################

add_gray_box <- function() {
  rect_df <- data.frame(xmin = -Inf, xmax = Inf, ymin = 0, ymax = 1e-6)

  geom_rect(data = rect_df,
            mapping = aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = "black", alpha = 0.15)
}


log10_with_0_trans <- function(constant = 10, n = 6) {
  breaks <- function(x) {
    b <- log_breaks(n = n, base = 10)(c(10^(-constant + 1), max(abs(x))))
    sort(c(0, b))
  }
  trans_new("log10_with_0_trans",
            transform = function(x) sign(x) * (log(10^(-constant) + abs(x), 10) + constant),
            inverse = function(y) sign(y) * (10^(sign(y) * y -constant) - 10^(-constant)),
            breaks = breaks)
}
