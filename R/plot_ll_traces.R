# ============================================================
# Trace plots for summed subject LL, population LL, and
# log-posterior (up to an additive constant).
# ============================================================

# ------------------------------------------------------------
# Internal chain-level extractors
# ------------------------------------------------------------

.get_summed_ll_chain <- function(chain) {
  colSums(chain$samples$subj_ll)
}

.get_pop_ll_chain <- function(chain) {
  alpha     <- chain$samples$alpha
  theta_mu  <- chain$samples$theta_mu
  theta_var <- chain$samples$theta_var

  n_iter <- dim(alpha)[3]
  out <- numeric(n_iter)
  for (i in seq_len(n_iter)) {
    out[i] <- sum(
      mvtnorm::dmvnorm(
        x     = t(alpha[, , i]),
        mean  = theta_mu[, i],
        sigma = theta_var[, , i],
        log   = TRUE
      )
    )
  }
  out
}

# ------------------------------------------------------------
# Internal prior-density helpers (log-posterior only)
# ------------------------------------------------------------

.dinvgamma_log <- function(x, shape, rate) {
  shape * log(rate) - lgamma(shape) - (shape + 1) * log(x) - rate / x
}

.diwish_log_unnorm <- function(Sigma, df, S) {
  p            <- nrow(Sigma)
  logdet_Sigma <- as.numeric(determinant(Sigma, logarithm = TRUE)$modulus)
  Sigma_inv    <- solve(Sigma)
  -0.5 * (df + p + 1) * logdet_Sigma - 0.5 * sum(diag(S %*% Sigma_inv))
}

.complete_standard_prior <- function(prior, p, mu_dim) {
  if (is.null(prior$theta_mu_mean)) prior$theta_mu_mean <- rep(0, mu_dim)
  if (is.null(prior$theta_mu_var))  prior$theta_mu_var  <- diag(mu_dim)
  if (is.null(prior$v))             prior$v <- 2
  if (is.null(prior$A))             prior$A <- rep(0.3, p)
  prior
}

.get_mu_prior_ll <- function(theta_mu, prior) {
  n_iter <- ncol(theta_mu)
  out    <- numeric(n_iter)
  for (i in seq_len(n_iter)) {
    out[i] <- mvtnorm::dmvnorm(
      x     = theta_mu[, i],
      mean  = prior$theta_mu_mean,
      sigma = prior$theta_mu_var,
      log   = TRUE
    )
  }
  out
}

.get_a_prior_ll <- function(a_half, prior) {
  n_iter <- ncol(a_half)
  out    <- numeric(n_iter)
  for (i in seq_len(n_iter)) {
    out[i] <- sum(
      .dinvgamma_log(x = a_half[, i], shape = 0.5, rate = 1 / (prior$A^2))
    )
  }
  out
}

.get_sigma_prior_ll <- function(theta_var, a_half, prior) {
  p      <- dim(theta_var)[1]
  n_iter <- dim(theta_var)[3]
  out    <- numeric(n_iter)
  for (i in seq_len(n_iter)) {
    S_i  <- 2 * prior$v * diag(1 / a_half[, i], p)
    df_i <- prior$v + p - 1
    out[i] <- .diwish_log_unnorm(Sigma = theta_var[, , i], df = df_i, S = S_i)
  }
  out
}

.get_post_ll_chain_standard <- function(chain, prior) {
  subj_ll   <- .get_summed_ll_chain(chain)
  pop_ll    <- .get_pop_ll_chain(chain)
  theta_mu  <- chain$samples$theta_mu
  theta_var <- chain$samples$theta_var
  a_half    <- chain$samples$a_half
  p         <- dim(theta_var)[1]

  prior <- .complete_standard_prior(prior, p = p, mu_dim = nrow(theta_mu))

  subj_ll +
    pop_ll +
    .get_mu_prior_ll(theta_mu, prior) +
    .get_a_prior_ll(a_half, prior) +
    .get_sigma_prior_ll(theta_var, a_half, prior)
}

# ------------------------------------------------------------
# Internal helpers: filtering, aligning, plotting
# ------------------------------------------------------------

.ll_filter_emc <- function(emc, stage, filter) {
  subset(emc, stage = stage, filter = filter, keep_stages = FALSE)
}

.is_hierarchical <- function(emc) {
  !is.null(emc[[1]]$samples$theta_mu)
}

.ll_align_to_min <- function(x) {
  min_iter <- min(vapply(x, length, integer(1)))
  lapply(x, function(v) v[seq_len(min_iter)])
}

.ll_as_matrix <- function(x) {
  x   <- .ll_align_to_min(x)
  mat <- do.call(cbind, x)
  colnames(mat) <- paste0("Chain_", seq_len(ncol(mat)))
  mat
}

.ll_align_two <- function(mat1, mat2) {
  n <- min(nrow(mat1), nrow(mat2))
  list(mat1 = mat1[seq_len(n), , drop = FALSE],
       mat2 = mat2[seq_len(n), , drop = FALSE])
}

.ll_summarise <- function(mat, use) {
  if (use == "mean")   return(rowMeans(mat, na.rm = TRUE))
  apply(mat, 1, stats::median, na.rm = TRUE)
}

.plot_ll_traces <- function(values_list, xlab, ylab, main) {
  cols  <- seq_along(values_list)
  y_all <- range(unlist(values_list), na.rm = TRUE)
  plot(values_list[[1]], type = "l", col = cols[1],
       ylim = y_all, xlab = xlab, ylab = ylab, main = main)
  if (length(values_list) > 1) {
    for (i in 2:length(values_list))
      graphics::lines(values_list[[i]], col = cols[i])
  }
  graphics::legend("bottomright",
                   legend = paste0("Chain_", seq_along(values_list)),
                   col = cols, lty = 1, bty = "n")
}

.plot_ll_diff <- function(diff_values, burn, xlab, ylab, main) {
  plot(diff_values, type = "l", xlab = xlab, ylab = ylab, main = main)
  graphics::abline(h = 0, lty = 2)
  if (burn > 0) graphics::abline(v = burn, lty = 3)
}

.ll_compute_diff <- function(worse_model, better_model, extractor, use) {
  worse_mat  <- .ll_as_matrix(lapply(worse_model, extractor))
  better_mat <- .ll_as_matrix(lapply(better_model, extractor))
  aligned    <- .ll_align_two(worse_mat, better_mat)
  list(
    diff       = .ll_summarise(aligned$mat2, use) - .ll_summarise(aligned$mat1, use),
    better_mat = aligned$mat2,
    worse_mat  = aligned$mat1
  )
}

# ------------------------------------------------------------
# Public: summed subject log-likelihood traces
# ------------------------------------------------------------

#' Trace Plot of Summed Subject Log-Likelihood
#'
#' Plots the summed (across subjects) log-likelihood trace for each chain.
#'
#' @param emc An emc object.
#' @param stage A character string. Stage to plot. Defaults to the last completed stage.
#' @param filter Integer or numeric vector passed to `subset.emc`.
#' @return Invisibly returns a matrix (iterations x chains) of summed log-likelihood values.
#' @export
plot_summed_ll <- function(emc, stage = get_last_stage(emc), filter = NULL) {
  emc  <- .ll_filter_emc(emc, stage, filter)
  vals <- lapply(emc, .get_summed_ll_chain)
  .plot_ll_traces(vals,
                  xlab = "Iteration",
                  ylab = "Summed subject log-likelihood",
                  main = "Summed subject log-likelihood trace")
  invisible(.ll_as_matrix(vals))
}

#' Trace Plot of Summed Subject Log-Likelihood Difference Between Two Models
#'
#' Plots the difference in chain-averaged summed log-likelihood between two models.
#'
#' @param worse_model An emc object for the worse (baseline) model.
#' @param better_model An emc object for the better model.
#' @param stage A character string. Stage to use. Defaults to `"sample"`.
#' @param filter Integer or numeric vector passed to `subset.emc`.
#' @param burn Integer. If > 0, draws a vertical line at this iteration.
#' @param use Character. Summary across chains: `"mean"` (default) or `"median"`.
#' @return Invisibly returns a list with `diff`, `better_mat`, and `worse_mat`.
#' @export
plot_summed_ll_diff <- function(worse_model, better_model,
                                stage = "sample", filter = NULL,
                                burn = 0, use = c("mean", "median")) {
  use          <- match.arg(use)
  worse_model  <- .ll_filter_emc(worse_model,  stage, filter)
  better_model <- .ll_filter_emc(better_model, stage, filter)
  result       <- .ll_compute_diff(worse_model, better_model, .get_summed_ll_chain, use)
  .plot_ll_diff(result$diff, burn,
                xlab = "Iteration",
                ylab = sprintf("Summed LL difference (%s; better - worse)", use),
                main = "Summed subject LL difference trace")
  invisible(result)
}

# ------------------------------------------------------------
# Public: population log-likelihood traces
# ------------------------------------------------------------

#' Trace Plot of Population Log-Likelihood
#'
#' Plots the population-level log-likelihood (i.e., the log-density of individual
#' parameters under the group distribution) trace for each chain.
#' Only applicable to hierarchical models.
#'
#' @inheritParams plot_summed_ll
#' @return Invisibly returns a matrix (iterations x chains) of population log-likelihood values.
#' @export
plot_pop_ll <- function(emc, stage = get_last_stage(emc), filter = NULL) {
  if (!.is_hierarchical(emc)) stop("plot_pop_ll requires a hierarchical model (type != 'single').")
  emc  <- .ll_filter_emc(emc, stage, filter)
  vals <- lapply(emc, .get_pop_ll_chain)
  .plot_ll_traces(vals,
                  xlab = "Iteration",
                  ylab = "Population log-likelihood",
                  main = "Population log-likelihood trace")
  invisible(.ll_as_matrix(vals))
}

#' Trace Plot of Population Log-Likelihood Difference Between Two Models
#'
#' @inheritParams plot_summed_ll_diff
#' @return Invisibly returns a list with `diff`, `better_mat`, and `worse_mat`.
#' @export
plot_pop_ll_diff <- function(worse_model, better_model,
                             stage = "sample", filter = NULL,
                             burn = 0, use = c("mean", "median")) {
  if (!.is_hierarchical(worse_model) || !.is_hierarchical(better_model))
    stop("plot_pop_ll_diff requires hierarchical models (type != 'single').")
  use          <- match.arg(use)
  worse_model  <- .ll_filter_emc(worse_model,  stage, filter)
  better_model <- .ll_filter_emc(better_model, stage, filter)
  result       <- .ll_compute_diff(worse_model, better_model, .get_pop_ll_chain, use)
  .plot_ll_diff(result$diff, burn,
                xlab = "Iteration",
                ylab = sprintf("Population LL difference (%s; better - worse)", use),
                main = "Population LL difference trace")
  invisible(result)
}

# ------------------------------------------------------------
# Public: log-posterior traces (standard variant only)
# ------------------------------------------------------------

#' Trace Plot of Log-Posterior
#'
#' Plots the log-posterior (up to an additive constant) trace for each chain.
#' Combines summed subject LL, population LL, and prior densities for
#' `theta_mu`, `theta_var`, and `a_half`. Only applicable to the standard
#' hierarchical variant.
#'
#' @inheritParams plot_summed_ll
#' @return Invisibly returns a matrix (iterations x chains) of log-posterior values.
#' @export
plot_post_ll <- function(emc, stage = get_last_stage(emc), filter = NULL) {
  if (!.is_hierarchical(emc)) stop("plot_post_ll requires a hierarchical model (type != 'single').")
  if (is.null(emc[[1]]$samples$a_half))
    stop("plot_post_ll currently only supports the standard variant (requires a_half).")
  emc   <- .ll_filter_emc(emc, stage, filter)
  prior <- get_prior_standard(
    prior        = NULL,
    n_pars       = dim(emc[[1]]$samples$alpha)[1],
    sample       = FALSE,
    group_design = emc[[1]]$group_designs
  )
  vals <- lapply(emc, .get_post_ll_chain_standard, prior = prior)
  .plot_ll_traces(vals,
                  xlab = "Iteration",
                  ylab = "Log-posterior",
                  main = "Log-posterior trace")
  invisible(.ll_as_matrix(vals))
}

#' Trace Plot of Log-Posterior Difference Between Two Models
#'
#' @inheritParams plot_summed_ll_diff
#' @return Invisibly returns a list with `diff`, `better_mat`, and `worse_mat`.
#' @export
plot_post_ll_diff <- function(worse_model, better_model,
                              stage = "sample", filter = NULL,
                              burn = 0, use = c("mean", "median")) {
  for (m in list(worse_model, better_model)) {
    if (!.is_hierarchical(m))    stop("plot_post_ll_diff requires hierarchical models.")
    if (is.null(m[[1]]$samples$a_half))
      stop("plot_post_ll_diff currently only supports the standard variant (requires a_half).")
  }
  use          <- match.arg(use)
  worse_model  <- .ll_filter_emc(worse_model,  stage, filter)
  better_model <- .ll_filter_emc(better_model, stage, filter)

  make_extractor <- function(mod) {
    prior <- get_prior_standard(
      prior        = NULL,
      n_pars       = dim(mod[[1]]$samples$alpha)[1],
      sample       = FALSE,
      group_design = mod[[1]]$group_designs
    )
    function(chain) .get_post_ll_chain_standard(chain, prior)
  }

  worse_mat  <- .ll_as_matrix(lapply(worse_model,  make_extractor(worse_model)))
  better_mat <- .ll_as_matrix(lapply(better_model, make_extractor(better_model)))
  aligned    <- .ll_align_two(worse_mat, better_mat)
  diff       <- .ll_summarise(aligned$mat2, use) - .ll_summarise(aligned$mat1, use)

  .plot_ll_diff(diff, burn,
                xlab = "Iteration",
                ylab = sprintf("Log-posterior difference (%s; better - worse)", use),
                main = "Log-posterior difference trace")
  invisible(list(diff = diff, better_mat = aligned$mat2, worse_mat = aligned$mat1))
}
