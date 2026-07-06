# Win-all / win-one feature-accumulator RDM models.
# Ported from EMC2-oo_refactor_simd_winall/R/model_WINALL_RDM.R (winall + winone only;
# winmix / winrepeat dropped). Additive: no existing model is modified.
#
# Each option has n_feat feature accumulators (Wald/RDM). lR levels:
#   "opt1_f1", "opt1_f2", "opt2_f1", "opt2_f2"  (opt1_* -> R level 1, opt2_* -> R level 2).
# Win-all: an option wins when its LAST feature crosses first. Win-one: its FIRST feature.
# Variable n_feat: Single trials -> 2 accumulator rows, Double -> 4. The n_acc_per_trial
# attribute (set by expand_accumulators_winall) is read by the R and C++ likelihoods.
# type = "WINALL" (drives add_accumulators expansion); c_name selects the C++ likelihood.

# ---- R log_likelihood fallbacks (mirror the C++) ---------------------------
log_likelihood_winall_R <- function(pars, dadm, model, min_ll = log(1e-10)) {
  # Reconstruct n_acc_per_trial + identity expand from the lR anchors (rows where the
  # lR factor == level 1 = "opt1_f1"); robust to the attribute not surviving per-subject
  # subsetting (mirrors the C++). compress = FALSE for these models.
  anchor <- which(as.integer(dadm$lR) == 1L)
  if (length(anchor) == 0L) stop("WINALL_RDM: no anchor rows (lR level 1) found")
  n_acc_per_trial <- diff(c(anchor, nrow(dadm) + 1L))
  expand <- seq_along(n_acc_per_trial)
  n_comp_trials <- length(n_acc_per_trial)

  d_all  <- model$dfun(dadm$rt, pars)
  p_all  <- model$pfun(dadm$rt, pars)
  winner <- dadm$winner
  ok     <- if (is.null(attr(pars, "ok"))) rep(TRUE, nrow(dadm)) else attr(pars, "ok")

  ll_trial  <- numeric(n_comp_trials)
  row_start <- 1L
  for (t in seq_len(n_comp_trials)) {
    na  <- n_acc_per_trial[t]
    idx <- row_start:(row_start + na - 1L)
    row_start <- row_start + na
    if (!ok[idx[1L]]) { ll_trial[t] <- min_ll; next }

    w          <- winner[idx]
    p_t        <- p_all[idx]
    d_t        <- d_all[idx]
    prod_p_win <- prod(p_t[w])
    prod_p_los <- prod(p_t[!w])
    p_win      <- p_t[w]
    density_max <- sum(d_t[w] * ifelse(p_win > 1e-300, prod_p_win / p_win, 0))
    surv_max    <- 1 - prod_p_los
    ll          <- log(density_max) + log(surv_max)
    ll_trial[t] <- if (is.finite(ll)) ll else min_ll
  }
  sum(pmax(min_ll, ll_trial[expand]))
}

log_likelihood_winone_R <- function(pars, dadm, model, min_ll = log(1e-10)) {
  anchor <- which(as.integer(dadm$lR) == 1L)
  if (length(anchor) == 0L) stop("WINONE_RDM: no anchor rows (lR level 1) found")
  n_acc_per_trial <- diff(c(anchor, nrow(dadm) + 1L))
  expand <- seq_along(n_acc_per_trial)

  d_all  <- model$dfun(dadm$rt, pars)
  p_all  <- model$pfun(dadm$rt, pars)
  winner <- dadm$winner
  ok     <- if (is.null(attr(pars, "ok"))) rep(TRUE, nrow(dadm)) else attr(pars, "ok")

  ll_trial  <- numeric(length(n_acc_per_trial))
  row_start <- 1L
  for (t in seq_along(n_acc_per_trial)) {
    na  <- n_acc_per_trial[t]
    idx <- row_start:(row_start + na - 1L)
    row_start <- row_start + na
    if (!ok[idx[1L]]) { ll_trial[t] <- min_ll; next }

    w       <- winner[idx]
    p_t     <- p_all[idx]
    d_t     <- d_all[idx]
    surv_t  <- pmax(0, 1 - p_t)
    win_idx <- which(w)
    los_idx <- which(!w)
    density_min <- 0
    for (ii in win_idx) {
      other_win <- setdiff(win_idx, ii)
      density_min <- density_min + d_t[ii] * prod(surv_t[other_win])
    }
    surv_min_los <- prod(surv_t[los_idx])
    ll <- log(density_min) + log(surv_min_los)
    ll_trial[t] <- if (is.finite(ll)) ll else min_ll
  }
  sum(pmax(min_ll, ll_trial[expand]))
}

# ---- accumulator expansion (custom_expand hook target) ---------------------
# cfg_accumulator = TRUE (2026-07-06, win-all-3): Double trials get a THIRD
# accumulator per option (lR level f3) carrying the configural channel as its
# own race member instead of a coactive drift term. Backward-compatible: the
# default FALSE reproduces the original 4-level expansion byte-for-byte. The
# f3 NAME (digits after _f) is required by the winner-regex below and by the
# C++ team extraction. Only variable mode supports the flag. NB the rfun
# FeatureCount fallback in .winall_n_acc_vec assumes 4 rows on Double; the
# attr/lR paths (which always apply in practice) are team-size generic.
expand_accumulators_winall <- function(n_feat_per_option = "variable",
                                       cfg_accumulator = FALSE) {
  variable_mode <- identical(n_feat_per_option, "variable")
  if (cfg_accumulator && !variable_mode)
    stop("cfg_accumulator = TRUE requires n_feat_per_option = 'variable'")
  if (!variable_mode) {
    fixed_n_feat <- as.integer(n_feat_per_option)
    fixed_n_acc  <- 2L * fixed_n_feat
  }
  all_lR_levels <- if (cfg_accumulator)
    c("opt1_f1", "opt1_f2", "opt1_f3", "opt2_f1", "opt2_f2", "opt2_f3")
  else
    c("opt1_f1", "opt1_f2", "opt2_f1", "opt2_f2")

  function(data, matchfun = NULL, simulate = FALSE) {
    n_data <- nrow(data)
    if (variable_mode) {
      fc        <- as.character(data$FeatureCount)
      dbl_acc   <- if (cfg_accumulator) 6L else 4L
      n_acc_vec <- ifelse(fc == "Double", dbl_acc, 2L)
    } else {
      n_acc_vec <- rep(fixed_n_acc, n_data)
    }
    make_lR <- function(na) {
      nf <- na / 2L
      c(paste0("opt1_f", seq_len(nf)), paste0("opt2_f", seq_len(nf)))
    }
    row_idx <- unlist(mapply(rep, seq_len(n_data), n_acc_vec, SIMPLIFY = FALSE, USE.NAMES = FALSE))
    lR_vals <- unlist(lapply(n_acc_vec, make_lR), use.names = FALSE)

    datar <- data[row_idx, , drop = FALSE]
    datar$lR <- factor(lR_vals, levels = all_lR_levels)
    row.names(datar) <- NULL
    if (!simulate) {
      lR_opt       <- as.integer(sub("opt(\\d+)_f\\d+", "\\1", lR_vals))
      datar$winner <- (lR_opt == as.integer(datar$R))
    }
    attr(datar, "n_acc_per_trial") <- n_acc_vec
    datar
  }
}

# ---- simulation --------------------------------------------------------------
.winall_n_acc_vec <- function(data, pars, variable_mode, fixed_n_acc) {
  n_rows <- nrow(pars)
  n_acc_attr <- if (!is.null(data)) attr(data, "n_acc_per_trial") else NULL
  if (!is.null(n_acc_attr) && sum(n_acc_attr) == n_rows) {
    n_acc_vec <- as.integer(n_acc_attr)
  } else if (!is.null(data) && !is.null(data$lR) && nrow(data) == n_rows) {
    lR_fac    <- if (is.factor(data$lR)) data$lR else factor(data$lR)
    anchor    <- which(as.character(lR_fac) == levels(lR_fac)[1L])
    n_acc_vec <- diff(c(anchor, n_rows + 1L))
  } else if (variable_mode && !is.null(data) && "FeatureCount" %in% names(data)) {
    fc        <- as.character(data$FeatureCount)
    n_acc_vec <- ifelse(fc == "Double", 4L, 2L)
  } else {
    n_acc_t   <- if (variable_mode) 4L else fixed_n_acc
    n_acc_vec <- rep(n_acc_t, n_rows / n_acc_t)
  }
  if (length(n_acc_vec) == 0L || any(n_acc_vec <= 0L) || sum(n_acc_vec) != n_rows)
    stop("WIN feature RDM rfun could not align accumulator rows to trials")
  as.integer(n_acc_vec)
}

.winall_rfun <- function(is_winone) {
  function(n_feat_per_option = "variable") {
    variable_mode <- identical(n_feat_per_option, "variable")
    fixed_n_acc   <- if (!variable_mode) 2L * as.integer(n_feat_per_option) else NA_integer_
    function(data = NULL, pars) {
      n_rows    <- nrow(pars)
      n_acc_vec <- .winall_n_acc_vec(data, pars, variable_mode, fixed_n_acc)
      n_trial   <- length(n_acc_vec)

      if (any(dimnames(pars)[[2]] == "s"))
        pars[, c("A", "B", "v")] <- pars[, c("A", "B", "v")] / pars[, "s"]
      pars[, "B"][pars[, "B"] < 0] <- 0
      pars[, "A"][pars[, "A"] < 0] <- 0

      ok <- attr(pars, "ok")
      if (is.null(ok)) ok <- rep(TRUE, n_rows)
      finish     <- rep(Inf, n_rows)
      finish[ok] <- rWald(sum(ok), B = pars[ok, "B"], v = pars[ok, "v"], A = pars[ok, "A"])
      abs_times  <- finish + pars[, "t0"]

      winner_idx <- integer(n_trial)
      rt_vec     <- numeric(n_trial)
      pars_row   <- 1L
      for (i in seq_len(n_trial)) {
        na   <- n_acc_vec[i]; nf <- na / 2L
        opt1 <- pars_row:(pars_row + nf - 1L)
        opt2 <- (pars_row + nf):(pars_row + na - 1L)
        if (is_winone) {
          t1 <- min(abs_times[opt1]); t2 <- min(abs_times[opt2])
        } else {
          t1 <- max(abs_times[opt1]); t2 <- max(abs_times[opt2])
        }
        winner_idx[i] <- if (t1 <= t2) 1L else 2L
        rt_vec[i]     <- min(t1, t2)
        pars_row      <- pars_row + na
      }
      R_levels <- levels(data$R)
      data.frame(R = factor(R_levels[winner_idx], levels = R_levels), rt = rt_vec)
    }
  }
}
rfun_winall <- .winall_rfun(is_winone = FALSE)
rfun_winone <- .winall_rfun(is_winone = TRUE)

# ---- model factories --------------------------------------------------------
.winall_model <- function(n_feat, c_name, rfun_maker, ll_R, cfg_accumulator = FALSE) {
  list(
    type    = "WINALL",
    c_name  = c_name,
    p_types = c(v = log(1), B = log(1), A = log(0), t0 = log(0), s = log(1)),
    transform = list(func = c(v = "exp", B = "exp", A = "exp", t0 = "exp", s = "exp")),
    bound = list(
      minmax    = cbind(v = c(1e-3, Inf), B = c(0, Inf), A = c(1e-4, Inf),
                        t0 = c(0.05, Inf), s = c(0, Inf)),
      exception = c(A = 0, v = 0)
    ),
    Ttransform = function(pars, dadm) cbind(pars, b = pars[, "B"] + pars[, "A"]),
    expand_accumulators = expand_accumulators_winall(n_feat, cfg_accumulator),
    rfun                = rfun_maker(n_feat),
    dfun                = function(rt, pars) dRDM(rt, pars),
    pfun                = function(rt, pars) pRDM(rt, pars),
    log_likelihood      = function(pars, dadm, model, min_ll = log(1e-10)) ll_R(pars, dadm, model, min_ll)
  )
}

#' Win-All Feature-Accumulator Model (RDM family)
#' @param n_feat Integer or "variable" (default: reads FeatureCount per trial).
#' @return A model list for use in \code{design()}.
#' @export
WINALL_RDM <- function(n_feat = "variable")
  .winall_model(n_feat, "WINALL_RDM", rfun_winall, log_likelihood_winall_R)

#' Win-All Model with a Dedicated Configural Accumulator (win-all-3)
#'
#' As \code{WINALL_RDM}, but Double trials expand to THREE accumulators per
#' option (lR levels f1/f2/f3): two feature accumulators plus a configural
#' accumulator that races as its own team member (the configural channel is no
#' longer a coactive drift term). Single trials are unchanged (one accumulator
#' per option). Likelihood/simulation are shared with WINALL_RDM (team-size
#' generic; c_name unchanged so the C++ dispatch is identical).
#' @param n_feat Must be "variable" (FeatureCount-driven expansion).
#' @return A model list for use in \code{design()}.
#' @export
WINALL3_RDM <- function(n_feat = "variable")
  .winall_model(n_feat, "WINALL_RDM", rfun_winall, log_likelihood_winall_R,
                cfg_accumulator = TRUE)

#' Win-One Feature-Accumulator Model (RDM family)
#' @param n_feat Integer or "variable" (default: reads FeatureCount per trial).
#' @return A model list for use in \code{design()}.
#' @export
WINONE_RDM <- function(n_feat = "variable")
  .winall_model(n_feat, "WINONE_RDM", rfun_winone, log_likelihood_winone_R)
