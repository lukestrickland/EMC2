#ifndef WINALL_H
#define WINALL_H

// Win-all / win-one feature-accumulator likelihoods for RDM distributions.
// Ported from EMC2-oo_refactor_simd_winall (src/winall.h + the lr_all_variable
// helper from src/utility_functions.h). Additive: no existing function is modified.
//
// Each trial has a variable number of contiguous rows (n_acc_per_trial[t]).
//   winner[i] == TRUE  -> feature i belongs to the chosen option's team.
//   winner[i] == FALSE -> feature i belongs to the unchosen option's team.
// Single trials (n_acc = 2): 1 feature per option -> reduces to standard RDM LL.
// Double trials (n_acc = 4): 2 features per option -> win-all / win-one formula.

#include "model_RDM.h"   // digt_core, pigt_core, clamp_a_l
#include "RaceSetup.h"   // RaceSpec, make_race_setup
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

// Variable-block-size bounds propagation: n_acc_vec[t] = accumulator rows for
// trial t (2 Single, 4 Double). If any row in a trial's block is out of bounds
// (0), the whole block is 0; NA propagates otherwise.
inline void lr_all_variable(std::vector<int>& ok, const int* n_acc_vec, int n_actual)
{
  int base = 0;
  int* x = ok.data();
  for (int t = 0; t < n_actual; ++t) {
    const int na = n_acc_vec[t];
    int state = 1;
    for (int j = 0; j < na; ++j) {
      const int v = x[base + j];
      if (v == 0) { state = 0; break; }
      if (v == NA_LOGICAL) state = NA_LOGICAL;
    }
    std::fill(x + base, x + base + na, state);
    base += na;
  }
}

// Computes dfun AND pfun for every row in one pass (no winner-flag skipping).
inline void fill_rdm_dp_all(
    const NumericVector& rts,
    const ParamTable&    pt,
    const RaceSpec&      spec,
    std::vector<double>& d_all,
    std::vector<double>& p_all)
{
  const int N  = rts.size();
  const double* rt = rts.begin();
  const double* v  = &pt.base(0, spec.col_v);
  const double* B  = &pt.base(0, spec.col_B);
  const double* A  = &pt.base(0, spec.col_A);
  const double* t0 = &pt.base(0, spec.col_t0);
  const double* s  = &pt.base(0, spec.col_s);

  for (int i = 0; i < N; ++i) {
    const double t_eff = rt[i] - t0[i];
    if (t_eff <= 0.0) { d_all[i] = 0.0; p_all[i] = 0.0; continue; }
    const double inv_s = 1.0 / s[i];
    double a = 0.5  * A[i] * inv_s;
    double l = v[i]       * inv_s;
    double k = B[i]       * inv_s + a;
    clamp_a_l(a, l);
    d_all[i] = digt_core(t_eff, k, l, a);
    p_all[i] = pigt_core(t_eff, k, l, a);
  }
}

// WIN-ALL: an option wins when its SLOWEST feature finishes before the loser's
// slowest. density_max_win = sum_i[ d_i * prod_{j!=i, same team} p_j ];
// surv_max_los = 1 - prod_j p_j (losing team).
inline double c_log_likelihood_winall_rdm(
    const ParamTable&        pt,
    const RaceSpec&          spec,
    const NumericVector&     rts,
    const LogicalVector&     winner,
    const std::vector<int>&  is_ok,
    const IntegerVector&     expand,
    double                   min_ll,
    const IntegerVector&     n_acc_vec,
    NumericVector&           ll_trial)
{
  const int n_rows   = rts.size();
  const int n_actual = n_acc_vec.size();
  if (winner.size() != n_rows)
    Rcpp::stop("WINALL_RDM winner column length must equal the expanded data rows");
  if (static_cast<int>(is_ok.size()) != n_rows)
    Rcpp::stop("WINALL_RDM bounds vector length must equal the expanded data rows");
  if (ll_trial.size() != n_actual)
    Rcpp::stop("WINALL_RDM ll_trial scratch length must equal n_acc_per_trial length");
  if (expand.size() <= 0)
    Rcpp::stop("WINALL_RDM expand attribute must not be empty");

  int n_acc_total = 0;
  for (int t = 0; t < n_actual; ++t) {
    if (n_acc_vec[t] <= 0) Rcpp::stop("WINALL_RDM n_acc_per_trial entries must be positive");
    n_acc_total += n_acc_vec[t];
  }
  if (n_acc_total != n_rows)
    Rcpp::stop("WINALL_RDM n_acc_per_trial mismatch");

  std::vector<double> d_all(n_rows), p_all(n_rows);
  fill_rdm_dp_all(rts, pt, spec, d_all, p_all);
  const int* win_flag = LOGICAL(winner);

  int base = 0;
  for (int t = 0; t < n_actual; ++t) {
    const int na = n_acc_vec[t];
    if (is_ok[base] != 1) { ll_trial[t] = min_ll; base += na; continue; }

    double prod_p_win = 1.0, prod_p_los = 1.0;
    for (int k = 0; k < na; ++k) {
      const int i = base + k;
      if (win_flag[i]) prod_p_win *= p_all[i];
      else             prod_p_los *= p_all[i];
    }
    double density_max = 0.0;
    for (int k = 0; k < na; ++k) {
      const int i = base + k;
      if (win_flag[i] && p_all[i] > 1e-300)
        density_max += d_all[i] * (prod_p_win / p_all[i]);
    }
    const double surv_max = 1.0 - prod_p_los;
    const double ll = std::log(density_max) + std::log(surv_max);
    ll_trial[t] = std::isfinite(ll) ? ll : min_ll;
    base += na;
  }

  double total = 0.0;
  for (int t = 0; t < expand.size(); ++t) {
    const int idx = expand[t] - 1;
    if (idx < 0 || idx >= n_actual) Rcpp::stop("WINALL_RDM expand index out of range");
    total += ll_trial[idx];
  }
  return total;
}

// WIN-ONE: an option wins when its FASTEST feature finishes before the loser's
// fastest. density_min_win = sum_i[ d_i * prod_{j!=i, same team} (1-p_j) ];
// surv_min_los = prod_j (1-p_j) (losing team).
inline double c_log_likelihood_winone_rdm(
    const ParamTable&        pt,
    const RaceSpec&          spec,
    const NumericVector&     rts,
    const LogicalVector&     winner,
    const std::vector<int>&  is_ok,
    const IntegerVector&     expand,
    double                   min_ll,
    const IntegerVector&     n_acc_vec,
    NumericVector&           ll_trial)
{
  const int n_rows   = rts.size();
  const int n_actual = n_acc_vec.size();
  if (winner.size() != n_rows)
    Rcpp::stop("WINONE_RDM winner column length must equal the expanded data rows");
  if (static_cast<int>(is_ok.size()) != n_rows)
    Rcpp::stop("WINONE_RDM bounds vector length must equal the expanded data rows");
  if (ll_trial.size() != n_actual)
    Rcpp::stop("WINONE_RDM ll_trial scratch length must equal n_acc_per_trial length");
  if (expand.size() <= 0)
    Rcpp::stop("WINONE_RDM expand attribute must not be empty");

  int n_acc_total = 0;
  for (int t = 0; t < n_actual; ++t) {
    if (n_acc_vec[t] <= 0) Rcpp::stop("WINONE_RDM n_acc_per_trial entries must be positive");
    n_acc_total += n_acc_vec[t];
  }
  if (n_acc_total != n_rows)
    Rcpp::stop("WINONE_RDM n_acc_per_trial mismatch");

  std::vector<double> d_all(n_rows), p_all(n_rows);
  fill_rdm_dp_all(rts, pt, spec, d_all, p_all);
  const int* win_flag = LOGICAL(winner);

  int base = 0;
  for (int t = 0; t < n_actual; ++t) {
    const int na = n_acc_vec[t];
    if (is_ok[base] != 1) { ll_trial[t] = min_ll; base += na; continue; }

    double density_min = 0.0;
    double surv_min_los = 1.0;
    for (int k = 0; k < na; ++k) {
      const int i = base + k;
      const double surv_i = std::max(0.0, 1.0 - p_all[i]);
      if (!win_flag[i]) surv_min_los *= surv_i;
    }
    for (int k = 0; k < na; ++k) {
      const int i = base + k;
      if (!win_flag[i]) continue;
      double other_win_surv = 1.0;
      for (int j = 0; j < na; ++j) {
        const int jj = base + j;
        if (jj == i || !win_flag[jj]) continue;
        other_win_surv *= std::max(0.0, 1.0 - p_all[jj]);
      }
      density_min += d_all[i] * other_win_surv;
    }
    const double ll = std::log(density_min) + std::log(surv_min_los);
    ll_trial[t] = std::isfinite(ll) ? ll : min_ll;
    base += na;
  }

  double total = 0.0;
  for (int t = 0; t < expand.size(); ++t) {
    const int idx = expand[t] - 1;
    if (idx < 0 || idx >= n_actual) Rcpp::stop("WINONE_RDM expand index out of range");
    total += ll_trial[idx];
  }
  return total;
}

#endif // WINALL_H
