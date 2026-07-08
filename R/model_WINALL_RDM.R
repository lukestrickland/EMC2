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
                                       cfg_accumulator = FALSE,
                                       persev_accumulator = FALSE) {
  # persev_accumulator = TRUE (2026-07-07, win-all-3 + choose-stay member):
  # Double teams additionally race a PERSEVERATION accumulator, lR level f9
  # (digits required by the winner regex; f9 avoids colliding with value-dim
  # levels). Single trials unchanged. Requires cfg_accumulator (built for the
  # win-all-3 architecture).
  variable_mode <- identical(n_feat_per_option, "variable")
  if (cfg_accumulator && !variable_mode)
    stop("cfg_accumulator = TRUE requires n_feat_per_option = 'variable'")
  if (persev_accumulator && !cfg_accumulator)
    stop("persev_accumulator = TRUE requires cfg_accumulator = TRUE")
  if (!variable_mode) {
    fixed_n_feat <- as.integer(n_feat_per_option)
    fixed_n_acc  <- 2L * fixed_n_feat
  }
  all_lR_levels <- if (persev_accumulator)
    c("opt1_f1", "opt1_f2", "opt1_f3", "opt1_f9",
      "opt2_f1", "opt2_f2", "opt2_f3", "opt2_f9")
  else if (cfg_accumulator)
    c("opt1_f1", "opt1_f2", "opt1_f3", "opt2_f1", "opt2_f2", "opt2_f3")
  else
    c("opt1_f1", "opt1_f2", "opt2_f1", "opt2_f2")

  function(data, matchfun = NULL, simulate = FALSE) {
    n_data <- nrow(data)
    if (variable_mode) {
      fc        <- as.character(data$FeatureCount)
      dbl_acc   <- if (persev_accumulator) 8L else if (cfg_accumulator) 6L else 4L
      n_acc_vec <- ifelse(fc == "Double", dbl_acc, 2L)
    } else {
      n_acc_vec <- rep(fixed_n_acc, n_data)
    }
    make_lR <- function(na) {
      if (na == 8L) {
        one <- c("f1", "f2", "f3", "f9")
        return(c(paste0("opt1_", one), paste0("opt2_", one)))
      }
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
.winall_model <- function(n_feat, c_name, rfun_maker, ll_R, cfg_accumulator = FALSE,
                          persev_accumulator = FALSE) {
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
    expand_accumulators = expand_accumulators_winall(n_feat, cfg_accumulator,
                                                     persev_accumulator),
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

#' Win-All-3 + choose-stay accumulator (RDM family)
#'
#' Win-all-3 (configural accumulator, lR f3) with an additional PERSEVERATION
#' accumulator per option on Double trials (lR f9): the previously-chosen
#' option's member idles fast (does not bind) while the switch team's member
#' runs at baseline (binds) -- choice bias without repeat speed-up. Single
#' trials unchanged.
#' @param n_feat Integer or "variable" (default: reads FeatureCount per trial).
#' @return A model list for use in \code{design()}.
#' @export
WINALL3P_RDM <- function(n_feat = "variable")
  .winall_model(n_feat, "WINALL_RDM", rfun_winall, log_likelihood_winall_R,
                cfg_accumulator = TRUE, persev_accumulator = TRUE)

#' Win-One Feature-Accumulator Model (RDM family)
#' @param n_feat Integer or "variable" (default: reads FeatureCount per trial).
#' @return A model list for use in \code{design()}.
#' @export
WINONE_RDM <- function(n_feat = "variable")
  .winall_model(n_feat, "WINONE_RDM", rfun_winone, log_likelihood_winone_R)

# ---- win-all-3 PERSEVERATION variants (2026-07-07) ---------------------------
# Both keep the win-all-3 race and add a NON-accumulation-coupled repeat
# process driven by the persev_own dadm column (row-level indicator of the
# previously-chosen option; all-zero on first encounters):
#   MIX: discrete mixture -- with prob p_rep the response comes from a LONE
#        repeat accumulator (same B/A/t0/s as the trial's anchor row, own
#        drift v_rep): L = (1-p)*L_race + p*1[R==prev]*d_rep(t).
#   OVR: first-past-the-post override racer (own drift v_ovr) racing the
#        whole race: L = L_race*(1-P_ovr) + 1[R==prev]*d_ovr*S_race, with
#        S_race = (1-prod_p_win)(1-prod_p_los) (neither option complete).
#   SWAP: label-capture mixture -- with prob p_rep the race runs as normal but
#        the emitted response LABEL is the previous within-pair choice; the RT
#        is the race's own finish time regardless of which option won:
#        L = (1-p)*L_race + p*1[R==prev]*f_race(t), f_race = the race's
#        MARGINAL finish density (observed-response likelihood + the exact
#        role swap). No extra accumulator, no v_* parameter: the repeat
#        process is RT-silent by construction (port of the flat-API
#        RDM_REPEAT_CHOICE design to win-all-3).
# Nesting: p_rep -> 0 / v_ovr -> 0 recover the plain win-all-3 likelihood.
# C++ twins: c_log_likelihood_winall3_persev_rdm (src/winall.h), dispatched by
# c_name WINALL3MIX_RDM / WINALL3OVR_RDM / WINALL3SWAP_RDM.
log_likelihood_winall3_persev_R <- function(mode) {
  mode <- match.arg(mode, c("mix", "ovr", "swap"))
  function(pars, dadm, model, min_ll = log(1e-10)) {
    anchor <- which(as.integer(dadm$lR) == 1L)
    if (length(anchor) == 0L) stop("WINALL3 persev variant: no anchor rows found")
    n_acc_per_trial <- diff(c(anchor, nrow(dadm) + 1L))
    n_comp <- length(n_acc_per_trial)

    d_all  <- model$dfun(dadm$rt, pars)
    p_all  <- model$pfun(dadm$rt, pars)
    winner <- dadm$winner
    persev <- dadm$persev_own
    ok     <- if (is.null(attr(pars, "ok"))) rep(TRUE, nrow(dadm)) else attr(pars, "ok")

    # lone/override accumulator: anchor-row pars with v replaced by the extra
    # drift. The swap variant has NO extra accumulator -- skip entirely.
    if (mode != "swap") {
      pars_x <- pars[anchor, , drop = FALSE]
      pars_x[, "v"] <- pars_x[, if (mode == "mix") "v_rep" else "v_ovr"]
      d_x <- model$dfun(dadm$rt[anchor], pars_x)
      p_x <- model$pfun(dadm$rt[anchor], pars_x)
    }

    ll_trial  <- numeric(n_comp)
    row_start <- 1L
    for (t in seq_len(n_comp)) {
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
      L_race      <- density_max * (1 - prod_p_los)

      pv         <- persev[idx]
      has_prev   <- any(pv > 0.5)
      rep_chosen <- any(pv > 0.5 & w)
      L <- L_race
      if (has_prev) {
        if (mode == "mix") {
          pr <- pars[idx[1L], "p_rep"]
          L  <- (1 - pr) * L_race + if (rep_chosen) pr * d_x[t] else 0
        } else if (mode == "swap") {
          p_los <- p_t[!w]
          density_max_los <- sum(d_t[!w] * ifelse(p_los > 1e-300, prod_p_los / p_los, 0))
          f_race <- L_race + density_max_los * (1 - prod_p_win)
          pr <- pars[idx[1L], "p_rep"]
          L  <- (1 - pr) * L_race + if (rep_chosen) pr * f_race else 0
        } else {
          S_race <- (1 - prod_p_win) * (1 - prod_p_los)
          L      <- L_race * (1 - p_x[t]) + if (rep_chosen) d_x[t] * S_race else 0
        }
      }
      ll <- log(L)
      ll_trial[t] <- if (is.finite(ll)) ll else min_ll
    }
    sum(pmax(min_ll, ll_trial))
  }
}

# Simulation: base win-all-3 race sim plus the repeat process. The extra drifts
# (mix/ovr) are divided by s exactly as the member drifts are; the repeat/
# override accumulator borrows the trial's scaled B/A and raw t0 from the
# anchor row. The swap variant draws NO extra finishing time: with prob p_rep
# the response label is overwritten with the previous within-pair choice and
# the race's RT is kept unchanged.
.winall3_persev_rfun <- function(mode) {
  mode <- match.arg(mode, c("mix", "ovr", "swap"))
  function(n_feat_per_option = "variable") {
    variable_mode <- identical(n_feat_per_option, "variable")
    fixed_n_acc   <- if (!variable_mode) 2L * as.integer(n_feat_per_option) else NA_integer_
    function(data = NULL, pars) {
      n_rows    <- nrow(pars)
      n_acc_vec <- .winall_n_acc_vec(data, pars, variable_mode, fixed_n_acc)
      n_trial   <- length(n_acc_vec)

      s_col <- if (any(dimnames(pars)[[2]] == "s")) pars[, "s"] else rep(1, n_rows)
      pars_adj <- pars
      pars_adj[, c("A", "B", "v")] <- pars_adj[, c("A", "B", "v")] / s_col
      pars_adj[, "B"][pars_adj[, "B"] < 0] <- 0
      pars_adj[, "A"][pars_adj[, "A"] < 0] <- 0

      ok <- attr(pars, "ok")
      if (is.null(ok)) ok <- rep(TRUE, n_rows)
      finish     <- rep(Inf, n_rows)
      finish[ok] <- rWald(sum(ok), B = pars_adj[ok, "B"],
                          v = pars_adj[ok, "v"], A = pars_adj[ok, "A"])
      abs_times  <- finish + pars[, "t0"]

      persev <- data$persev_own
      if (is.null(persev)) stop("WINALL3 persev variant rfun: data lacks persev_own")

      winner_idx <- integer(n_trial)
      rt_vec     <- numeric(n_trial)
      pars_row   <- 1L
      for (i in seq_len(n_trial)) {
        na   <- n_acc_vec[i]; nf <- na / 2L
        idx  <- pars_row:(pars_row + na - 1L)
        opt1 <- idx[seq_len(nf)]
        opt2 <- idx[(nf + 1L):na]
        t1 <- max(abs_times[opt1]); t2 <- max(abs_times[opt2])
        R_i  <- if (t1 <= t2) 1L else 2L
        rt_i <- min(t1, t2)

        pv <- persev[idx]
        if (any(pv > 0.5)) {
          prev_opt <- if (any(pv[seq_len(nf)] > 0.5)) 1L else 2L
          a0 <- idx[1L]
          if (mode == "swap") {
            # label capture: response overwritten, race RT untouched
            if (stats::runif(1) < pars[a0, "p_rep"]) R_i <- prev_opt
          } else {
            v_x <- pars[a0, if (mode == "mix") "v_rep" else "v_ovr"] / s_col[a0]
            if (mode == "mix") {
              if (stats::runif(1) < pars[a0, "p_rep"]) {
                R_i  <- prev_opt
                rt_i <- pars[a0, "t0"] +
                  rWald(1, B = pars_adj[a0, "B"], v = v_x, A = pars_adj[a0, "A"])
              }
            } else {
              t_ovr <- pars[a0, "t0"] +
                rWald(1, B = pars_adj[a0, "B"], v = v_x, A = pars_adj[a0, "A"])
              if (t_ovr < rt_i) { R_i <- prev_opt; rt_i <- t_ovr }
            }
          }
        }
        winner_idx[i] <- R_i
        rt_vec[i]     <- rt_i
        pars_row      <- pars_row + na
      }
      R_levels <- levels(data$R)
      data.frame(R = factor(R_levels[winner_idx], levels = R_levels), rt = rt_vec)
    }
  }
}

.winall3_persev_variant_list <- function(n_feat, mode, cfg_accumulator = TRUE) {
  mode <- match.arg(mode, c("mix", "ovr", "swap"))
  if (!cfg_accumulator && mode != "swap")
    stop("cfg_accumulator = FALSE is only supported for the swap variant")
  # The C++ likelihood (c_log_likelihood_winall3_persev_rdm) is team-size
  # generic -- it reconstructs per-trial accumulator counts from the lR
  # anchors -- so the 2-leg (champion-architecture) swap reuses the installed
  # WINALL3SWAP_RDM dispatch string, exactly as WINALL3_RDM reuses the
  # WINALL_RDM c_name. Only the expansion differs (cfg_accumulator).
  c_name <- switch(mode,
                   mix  = "WINALL3MIX_RDM",
                   ovr  = "WINALL3OVR_RDM",
                   swap = "WINALL3SWAP_RDM")
  m <- .winall_model(n_feat,
                     c_name,
                     .winall3_persev_rfun(mode),
                     log_likelihood_winall3_persev_R(mode),
                     cfg_accumulator = cfg_accumulator)
  if (mode == "mix") {
    m$p_types <- c(m$p_types, v_rep = log(1), p_rep = stats::qnorm(0.1))
    m$transform$func <- c(m$transform$func, v_rep = "exp", p_rep = "pnorm")
  } else if (mode == "swap") {
    m$p_types <- c(m$p_types, p_rep = stats::qnorm(0.1))
    m$transform$func <- c(m$transform$func, p_rep = "pnorm")
  } else {
    m$p_types <- c(m$p_types, v_ovr = log(1))
    m$transform$func <- c(m$transform$func, v_ovr = "exp")
  }
  m
}

#' Win-All-3 + repeat MIXTURE (RDM family)
#' @param n_feat Integer or "variable" (default: reads FeatureCount per trial).
#' @return A model list for use in \code{design()}.
#' @export
WINALL3MIX_RDM <- function(n_feat = "variable")
  .winall3_persev_variant_list(n_feat, mode = "mix")

#' Win-All-3 + OVERRIDE repeat racer (RDM family)
#' @param n_feat Integer or "variable" (default: reads FeatureCount per trial).
#' @return A model list for use in \code{design()}.
#' @export
WINALL3OVR_RDM <- function(n_feat = "variable")
  .winall3_persev_variant_list(n_feat, mode = "ovr")

#' Win-All-3 + repeat LABEL-SWAP mixture (RDM family)
#'
#' Label-capture repeat mixture: with probability \code{p_rep} the race runs
#' exactly as normal but the emitted response is the previous within-pair
#' choice, with the RT taken from the race's own finish time regardless of
#' which option won (the marginal finish density in the likelihood). The
#' repeat process is therefore RT-silent by construction; \code{p_rep -> 0}
#' nests the plain win-all-3 race. No extra accumulator or drift parameter.
#' @param n_feat Integer or "variable" (default: reads FeatureCount per trial).
#' @return A model list for use in \code{design()}.
#' @export
WINALL3SWAP_RDM <- function(n_feat = "variable")
  .winall3_persev_variant_list(n_feat, mode = "swap")

#' Win-All + repeat LABEL-SWAP mixture (RDM family; champion 2-leg architecture)
#'
#' The label-swap repeat mixture of \code{WINALL3SWAP_RDM} on the plain
#' win-all expansion (no configural accumulator; 4 rows on Double, 2 on
#' Single) -- i.e. the capture process added on top of the champion
#' architecture, where configural evidence enters as a team-wide drift
#' contribution. Same likelihood dispatch (team-size generic); only the
#' expansion differs. \code{p_rep -> 0} nests the plain win-all race.
#' @param n_feat Integer or "variable" (default: reads FeatureCount per trial).
#' @return A model list for use in \code{design()}.
#' @export
WINALLSWAP_RDM <- function(n_feat = "variable")
  .winall3_persev_variant_list(n_feat, mode = "swap", cfg_accumulator = FALSE)
