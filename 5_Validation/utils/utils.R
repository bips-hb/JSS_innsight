
################################################################################
#                         Utility functions
################################################################################

add_gray_box <- function() {
  rect_df <- data.frame(xmin = -Inf, xmax = Inf, ymin = 0, ymax = 1e-6)

  geom_rect(data = rect_df,
            mapping = aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = "black", alpha = 0.15)
}


log10_with_0_trans <- function(constant = 10) {
  breaks <- function(x) {
    b <- log_breaks(n = 6, base = 10)(c(10^(-constant + 1), max(abs(x))))
    sort(c(0, b))
  }
  trans_new("log10_with_0_trans",
            transform = function(x) sign(x) * (log(10^(-constant) + abs(x), 10) + constant),
            inverse = function(y) sign(y) * (10^(sign(y) * y -constant) - 10^(-constant)),
            breaks = breaks)
}
