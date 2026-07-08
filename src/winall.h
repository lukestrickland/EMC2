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

// ---------------------------------------------------------------------------
// WIN-ALL-3 PERSEVERATION VARIANTS (2026-07-07). Both keep the win-all-3 race
// intact and add a NON-accumulation-coupled repeat process driven by the
// persev_own dadm column (1 on the previously-chosen option's rows; all-zero
// on first encounters). The repeat/override accumulator borrows the trial's
// B/A/t0/s from the ANCHOR row and takes its own drift from an extra
// ParamTable column (resolved by the dispatcher via base_index_for).
//
// MIX ("WINALL3MIX_RDM"): discrete mixture. With prob p_rep (column, pnorm-
//   transformed) the response is generated by a LONE repeat accumulator
//   (drift v_rep): L = (1-p)*L_race + p*1[R==prev]*d_rep(t). No prev: L_race.
// OVR ("WINALL3OVR_RDM"): first-past-the-post override racer (drift v_ovr)
//   racing the whole win-all race: L = L_race*(1-P_ovr(t))
//   + 1[R==prev]*d_ovr(t)*S_race(t), where S_race = (1-prod_p_win)(1-prod_p_los)
//   (neither option complete; options independent). No prev: L_race.
// SWAP ("WINALL3SWAP_RDM"): label-capture mixture (2026-07-07 evening). With
//   prob p_rep the race runs EXACTLY as normal but the emitted response LABEL
//   is the previous within-pair choice; the RT is the race's own finish time
//   REGARDLESS of which option won: L = (1-p)*L_race
//   + p*1[R==prev]*f_race(t), with f_race(t) = L_race(opt A wins, t)
//   + L_race(opt B wins, t) the race's MARGINAL finish density (the two win
//   events partition the finish event). No new RT process, no extra
//   accumulator, no v_* parameter -- the repeat process is RT-silent by
//   construction (port of the flat-API RDM_REPEAT_CHOICE design to win-all-3).
// Nesting: p_rep -> 0 and v_ovr -> 0 recover the win-all-3 likelihood.
// ---------------------------------------------------------------------------
inline void rdm_dp_single(double t_eff, double A, double B, double s, double v,
                          double& d_out, double& p_out)
{
  d_out = 0.0; p_out = 0.0;
  if (t_eff <= 0.0) return;
  const double inv_s = 1.0 / s;
  double a = 0.5 * A * inv_s;
  double l = v       * inv_s;
  double k = B       * inv_s + a;
  clamp_a_l(a, l);
  d_out = digt_core(t_eff, k, l, a);
  p_out = pigt_core(t_eff, k, l, a);
}

inline double c_log_likelihood_winall3_persev_rdm(
    const int                mode,          // 0 = MIX, 1 = OVR, 2 = SWAP
    const ParamTable&        pt,
    const RaceSpec&          spec,
    const int                col_v_extra,   // v_rep (mix) / v_ovr (ovr); -1 for swap
    const int                col_p_rep,     // mix/swap; -1 for ovr
    const NumericVector&     rts,
    const LogicalVector&     winner,
    const NumericVector&     persev,
    const std::vector<int>&  is_ok,
    const IntegerVector&     expand,
    double                   min_ll,
    const IntegerVector&     n_acc_vec,
    NumericVector&           ll_trial)
{
  const int n_rows   = rts.size();
  const int n_actual = n_acc_vec.size();
  if (winner.size() != n_rows || persev.size() != n_rows)
    Rcpp::stop("WINALL3 persev variant: winner/persev_own length must equal the expanded rows");
  if (static_cast<int>(is_ok.size()) != n_rows)
    Rcpp::stop("WINALL3 persev variant: bounds vector length must equal the expanded rows");
  if (ll_trial.size() != n_actual)
    Rcpp::stop("WINALL3 persev variant: ll_trial scratch length mismatch");

  int n_acc_total = 0;
  for (int t = 0; t < n_actual; ++t) {
    if (n_acc_vec[t] <= 0) Rcpp::stop("WINALL3 persev variant: non-positive n_acc entry");
    n_acc_total += n_acc_vec[t];
  }
  if (n_acc_total != n_rows)
    Rcpp::stop("WINALL3 persev variant: n_acc_per_trial mismatch");

  std::vector<double> d_all(n_rows), p_all(n_rows);
  fill_rdm_dp_all(rts, pt, spec, d_all, p_all);
  const int* win_flag = LOGICAL(winner);

  const double* A_col  = &pt.base(0, spec.col_A);
  const double* B_col  = &pt.base(0, spec.col_B);
  const double* t0_col = &pt.base(0, spec.col_t0);
  const double* s_col  = &pt.base(0, spec.col_s);
  const double* v_ext  = (col_v_extra >= 0) ? &pt.base(0, col_v_extra) : nullptr;
  const double* p_rep  = (col_p_rep   >= 0) ? &pt.base(0, col_p_rep)   : nullptr;

  int base = 0;
  for (int t = 0; t < n_actual; ++t) {
    const int na = n_acc_vec[t];
    if (is_ok[base] != 1) { ll_trial[t] = min_ll; base += na; continue; }

    double prod_p_win = 1.0, prod_p_los = 1.0;
    bool   has_prev = false, rep_chosen = false;
    for (int k = 0; k < na; ++k) {
      const int i = base + k;
      if (win_flag[i]) prod_p_win *= p_all[i];
      else             prod_p_los *= p_all[i];
      if (persev[i] > 0.5) { has_prev = true; if (win_flag[i]) rep_chosen = true; }
    }
    double density_max = 0.0;
    for (int k = 0; k < na; ++k) {
      const int i = base + k;
      if (win_flag[i] && p_all[i] > 1e-300)
        density_max += d_all[i] * (prod_p_win / p_all[i]);
    }
    const double L_race = density_max * (1.0 - prod_p_los);

    double L = L_race;
    if (has_prev) {
      const int i0 = base;
      if (mode == 2) {
        // SWAP: no new RT process. f_race = observed-response likelihood plus
        // the exact role swap (loser-team win density * winner-team survivor);
        // reuses the d/p values already computed for every row.
        double density_max_los = 0.0;
        for (int k = 0; k < na; ++k) {
          const int i = base + k;
          if (!win_flag[i] && p_all[i] > 1e-300)
            density_max_los += d_all[i] * (prod_p_los / p_all[i]);
        }
        const double f_race = L_race + density_max_los * (1.0 - prod_p_win);
        const double pr = p_rep[i0];
        L = (1.0 - pr) * L_race + (rep_chosen ? pr * f_race : 0.0);
      } else {
        const double t_eff = rts[i0] - t0_col[i0];
        double d_x = 0.0, p_x = 0.0;
        rdm_dp_single(t_eff, A_col[i0], B_col[i0], s_col[i0], v_ext[i0], d_x, p_x);
        if (mode == 0) {
          const double pr = p_rep[i0];
          L = (1.0 - pr) * L_race + (rep_chosen ? pr * d_x : 0.0);
        } else {
          const double S_race = (1.0 - prod_p_win) * (1.0 - prod_p_los);
          L = L_race * (1.0 - p_x) + (rep_chosen ? d_x * S_race : 0.0);
        }
      }
    }
    const double ll = std::log(L);
    ll_trial[t] = std::isfinite(ll) ? ll : min_ll;
    base += na;
  }

  double total = 0.0;
  for (int t = 0; t < expand.size(); ++t) {
    const int idx = expand[t] - 1;
    if (idx < 0 || idx >= n_actual) Rcpp::stop("WINALL3 persev variant: expand index out of range");
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
