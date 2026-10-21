# Flatness stopping rule (max_flat_loc) and sample-stage trimming (max_sample_iter)

test_that("flat location statistic behaves as defined", {
  set.seed(1)
  n <- 300
  mat <- cbind(stationary = rnorm(n),
               step = c(rep(0, n / 2), rep(5, n / 2)) + rnorm(n, sd = 0.1),
               constant = rep(1, n))
  s <- EMC2:::.flat_loc_stats_per_col(mat)
  expect_lt(s[["stationary"]], 0.25)
  expect_gt(s[["step"]], 0.9)
  expect_equal(s[["constant"]], 0)
  # Too few draws to judge: never flat
  expect_true(all(is.infinite(EMC2:::.flat_loc_stats_per_col(mat[1:9, ]))))
})

test_that("trimming keeps exactly the most recent sample draws", {
  emc <- samples_LNR
  n_before <- chain_n(emc)[1, "sample"]
  trimmed <- EMC2:::.trim_sample_stage(emc, 30)
  expect_true(all(chain_n(trimmed)[, "sample"] == 30))
  for (i in seq_along(emc)) {
    keep <- (n_before - 29):n_before
    expect_equal(trimmed[[i]]$samples$alpha, emc[[i]]$samples$alpha[, , keep, drop = FALSE])
    expect_equal(trimmed[[i]]$samples$subj_ll, emc[[i]]$samples$subj_ll[, keep, drop = FALSE])
    expect_equal(trimmed[[i]]$samples$idx, 30)
    expect_length(trimmed[[i]]$samples$stage, 30)
  }
  # No-op when already short enough
  expect_identical(EMC2:::.trim_sample_stage(emc, n_before), emc)
})

test_that("flatness check returns a usable verdict", {
  out <- EMC2:::.check_flatness_stop(samples_LNR, max_flat_loc = 0.25,
                                     flat_selection = c("alpha", "subj_ll", "theta_mu"))
  expect_true(is.finite(out$max_flat))
  expect_equal(out$flat_done, out$max_flat <= 0.25)
  expect_setequal(names(out$max_by_selection), c("alpha", "subj_ll", "theta_mu"))
  expect_true(EMC2:::.check_flatness_stop(samples_LNR, max_flat_loc = Inf,
                                          flat_selection = "alpha")$flat_done)
})

test_that("stop criteria are validated and defaulted", {
  sc <- EMC2:::get_stop_criteria("sample", list(max_flat_loc = 0.25), type = "standard")
  expect_equal(sc$flat_selection, c("alpha", "subj_ll", "theta_mu"))
  expect_equal(sc$flat_p1, 1/3)
  expect_equal(sc$flat_p2, 1/3)
  expect_error(EMC2:::get_stop_criteria("sample", list(max_flat_loc = 0.25, flat_p1 = 0), "standard"), "flat_p1")
  expect_error(EMC2:::get_stop_criteria("sample", list(max_flat_loc = 0.25, flat_selection = "mu"), "standard"), "flat_selection")
  expect_error(EMC2:::get_stop_criteria("sample", list(max_sample_iter = 5), "standard"), "max_sample_iter")
})

test_that("fit() resumes, trims to max_sample_iter and keeps going until flat", {
  skip_on_cran()
  # samples_LNR already holds 50 sample draws per chain, so this also exercises
  # resuming a fit. An unattainable flatness threshold keeps sampling going for
  # max_tries steps: 50 + 25 = 75 draws, trimmed back to the last 60 each step.
  set.seed(123)
  sc <- list(iter = 25, max_gd = 100, max_flat_loc = 1e-6, max_sample_iter = 60)
  expect_warning(
    emc <- fit(samples_LNR, stage = "sample", cores_for_chains = 1, stop_criteria = sc,
               verbose = FALSE, particle_factor = 30, step_size = 25, max_tries = 2),
    "flatness")
  expect_true(all(chain_n(emc)[, "sample"] == 60))

  # iter larger than max_sample_iter can never finish
  sc$iter <- 70
  expect_error(fit(samples_LNR, stage = "sample", cores_for_chains = 1, stop_criteria = sc,
                   verbose = FALSE), "exceeds max_sample_iter")
})
