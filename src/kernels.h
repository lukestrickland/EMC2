#ifndef KERNELS_H
#define KERNELS_H

#include <unordered_map>
#include <memory>
#include <array>
#include <vector>     //
#include <algorithm>  // std::fill, std::copy  (delta_satlink_gamma_card)
#include <cmath>      // std::pow, std::tanh, std::fabs
#include <Rcpp.h>    //
#include "nan_check.h"
#include "EMC2/userfun.hpp"
#include "Mat.h"
#include "kernels_math.h"

// View
struct KernelParsView {
  int n_rows;
  std::vector<const double*> cols;  // cols[k][row] = value for param k at trial row
};

// Struct for optional kernel arguments. Currently "q-value resetting", the DBM/TPM
// grid resolution, and the elemental feature-index columns + cardinality (n_elem) used
// by the cardinality-saturation read-out kernel (delta_satlink_gamma_card) are supported.
struct KernelArgs {
  const int* q_reset = nullptr;  // raw pointer into an IntegerVector; null = no reset
  int grid_res = 100;
  // delta_satlink_gamma_card: per-trial 1-based indices (into the n_elem covariate
  // columns) of each option's features; the *_2 columns are NA on single-item trials.
  const int* feat_one_1_idx = nullptr;
  const int* feat_one_2_idx = nullptr;
  const int* feat_two_1_idx = nullptr;
  const int* feat_two_2_idx = nullptr;
  // rw_satlink_gamma_card hybrid: OPTIONAL 3rd cue per option (the configural compound),
  // so the shared-PE read-out difference can span {feat_1, feat_2, compound}.
  const int* feat_one_3_idx = nullptr;
  const int* feat_two_3_idx = nullptr;
  int n_elem = -1;               // number of elemental (covariate) Q columns
  // delta_expdecr / delta_satlink_gamma_card_expdecr: within-block 0-based exposure
  // count driving the exp-decreasing learning rate alpha_eff(t).
  const double* block_trial = nullptr;
  // Future extensible fields go here, e.g.:
  // const double* some_other_col = nullptr;
};

// Struct for outputs. Outputs sometimes have 1 column, sometimes multiple. Pure C++ (for future threadsafe ops)
struct KernelOutput {
  const double* data = nullptr;  // raw pointer into kernel-owned storage
  int n_rows = 0;
  int n_cols = 1;                // 1 for all current kernels, N for RescorlaWagner

  // Convenience: element access (column-major, matches R matrix layout)
  double operator()(int r, int col) const {
    return data[col * n_rows + r];
  }
};

// ---- Types ----

enum class KernelType {
  SimpleDelta,
  Delta2Kernel,
  // Delta2Kernel2,
  Delta2LR,
  LinIncr,
  LinDecr,
  ExpIncr,
  ExpDecr,
  PowIncr,
  PowDecr,
  Poly2,
  Poly3,
  Poly4,
  Custom,
  RescorlaWagner,
  DeltaSatlinkGammaCard,
  DeltaSatlinkGammaCardExpdecr,
  DeltaSatlinkGammaCardPh,
  DeltaExpdecr,
  RwSatlinkGammaCard,
  RwSatlinkGammaCardFf,
  RwSatlinkGammaCardPh,
  RwSatlinkGammaCardExpdecr,
  BetaBinomial,
  BetaBinomialDecay,
  BetaBinomialWindow,
  DBM,
  TPM
};

// Some meta-data for kernels -- mostly for the future
struct KernelMeta {
  int  input_arity;          // how many *inputs* the kernel expects at once
  bool supports_grouping;    // whether a vector of names should be expanded into separate kernels
};

inline KernelMeta kernel_meta(KernelType kt) {
  switch (kt) {
  case KernelType::SimpleDelta:
  case KernelType::Delta2Kernel:
  // case KernelType::Delta2Kernel2:
  case KernelType::Delta2LR:
  case KernelType::LinIncr:
  case KernelType::LinDecr:
  case KernelType::ExpIncr:
  case KernelType::ExpDecr:
  case KernelType::PowIncr:
  case KernelType::PowDecr:
  case KernelType::Poly2:
  case KernelType::Poly3:
  case KernelType::Poly4:
  case KernelType::DeltaExpdecr:
    return {1, true};   // all current kernels: 1D input, grouping allowed
  case KernelType::Custom: return{1, false};
  case KernelType::RescorlaWagner: return{-1, false};  // N columns allowed
  case KernelType::DeltaSatlinkGammaCard:
  case KernelType::DeltaSatlinkGammaCardExpdecr:
  case KernelType::DeltaSatlinkGammaCardPh:
  case KernelType::RwSatlinkGammaCard:
  case KernelType::RwSatlinkGammaCardFf:
  case KernelType::RwSatlinkGammaCardPh:
  case KernelType::RwSatlinkGammaCardExpdecr: return{-1, false};  // variadic: all n_elem cols at once
  case KernelType::BetaBinomial:
  case KernelType::BetaBinomialDecay:
  case KernelType::BetaBinomialWindow:
  case KernelType::DBM:
  case KernelType::TPM:
    return {1, false};
  }

  // default future behaviour: 1D, but no grouping
  return {1, false};
}

// ---- Base + hierarchy ----
struct BaseKernel {
protected:
  std::vector<double> out_;
  bool has_run_ = false;

  // Remember expansion mapping (for 'at')
  std::vector<int> expand_idx_;   // 1-based indices
  bool has_expand_idx_ = false;

  // Two separate transpose buffers, one per stream family.
  // stream_buf_[0] is used for stream code 1 (Q / primary output)
  // stream_buf_[1] is used for stream code 2 (PE / secondary output)
  // Subclasses that need more streams can add further buffers explicitly.
  mutable std::vector<double> stream_buf_[2];

public:
  virtual ~BaseKernel() {}

  virtual void set_kernel_args(const KernelArgs& /*args*/) {}

  virtual void run(const KernelParsView& kernel_pars,
                   const Mat& covariate,
                   const std::vector<int>& comp_idx) = 0;

  virtual void reset() {
    out_.clear();
    stream_buf_[0].clear();
    stream_buf_[1].clear();
    has_run_ = false;
    expand_idx_.clear();
    has_expand_idx_ = false;
  }


  bool has_run() const { return has_run_; }

  // const std::vector<double>& get_output() const { return out_; }

  void set_expand_idx(const std::vector<int>& idx) {
    expand_idx_ = idx;
    has_expand_idx_ = !expand_idx_.empty();
  }
  const std::vector<int>& expand_idx() const { return expand_idx_; }
  bool has_expand_idx() const { return has_expand_idx_; }

  // Expand compressed out_ (length n_comp) into full length using expand_idx
  void do_expand(const std::vector<int>& expand_idx) {
    const int n_full = static_cast<int>(expand_idx.size());

    out_.resize(n_full);  // keeps compressed data in [0..n_comp-1]

    for (int i = n_full - 1; i >= 0; --i) {
      int k = expand_idx[i] - 1;  // 1-based -> 0-based
      out_[i] = out_[k];
    }
  }

  // does this kernel have a stream for this code?
  virtual bool has_output_stream(int code) const {
    return (code == 1);  // default: only main trajectory
  }

  // Returns a KernelOutput view into kernel-owned storage.
  // For single-column kernels: points directly into out_ (zero-copy).
  // For multi-column kernels (RescorlaWagner): points into col_major_buf_
  // which is populated lazily on first call.
  virtual KernelOutput get_output_stream(int code) const {
    if (code != 1) {
      Rcpp::stop("BaseKernel::get_output_stream: unsupported code %d (only 1)", code);
    }
    KernelOutput ko;
    ko.data   = out_.data();
    ko.n_rows = static_cast<int>(out_.size());
    ko.n_cols = 1;
    return ko;
  }

  // // single-stream getter, code=1 for main trajectory by default. code=2 for pes in delta, code=3 for xx in new kernels
  // virtual Rcpp::NumericVector get_output_stream(int code) const {
  //   using namespace Rcpp;
  //   if (code != 1) {
  //     stop("BaseKernel::get_output_stream: unsupported code %d (only 1)", code);
  //   }
  //   // out_ is already full-length at this point
  //   return wrap(out_);  // copies to NumericVector
  // }

  // Optional: name for each stream
  virtual std::string output_stream_name(int code) const {
    if (code == 1) return "covariate";
    throw std::runtime_error("BaseKernel::output_stream_name: unsupported code");
  }


protected:
  void mark_run_complete() { has_run_ = true; }
};

struct CustomKernel : BaseKernel {
private:
  Rcpp::XPtr<userfun_t> fun_;

public:
  // funptrSEXP is the external pointer stored in trend$custom_ptr
  CustomKernel(SEXP funptrSEXP) : fun_(funptrSEXP) {
    if (fun_.get() == nullptr) {
      Rcpp::stop("CustomKernel: null function pointer.");
    }
    if (!(*fun_)) {
      Rcpp::stop("CustomKernel: invalid function pointer.");
    }
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& input,
           const std::vector<int>& comp_idx) override {

             const int n_comp   = static_cast<int>(comp_idx.size());
             const int n_pars   = static_cast<int>(kernel_pars.cols.size());
             const int n_inputs = input.ncol;

             if (n_comp == 0) {
               out_.clear();
               mark_run_complete();
               return;
             }

             // 1) Build compressed parameter matrix: n_comp x n_pars
             Rcpp::NumericMatrix pars_comp(n_comp, n_pars);
             for (int p = 0; p < n_pars; ++p) {
               const double* col = kernel_pars.cols[p];
               for (int j = 0; j < n_comp; ++j) {
                 int r = comp_idx[j];        // full trial index
                 pars_comp(j, p) = col[r];   // compressed row j
               }
             }

             // 2) Build compressed input matrix: n_comp x n_inputs
             Rcpp::NumericMatrix input_comp(n_comp, n_inputs);
             for (int j = 0; j < n_comp; ++j) {
               int r = comp_idx[j];
               for (int c = 0; c < n_inputs; ++c) {
                 input_comp(j, c) = input(r, c);
               }
             }

             // 3) Call user function
             userfun_t f = *fun_;
             Rcpp::NumericVector res = f(pars_comp, input_comp);

             if (res.size() != n_comp) {
               Rcpp::stop("CustomKernel: user function returned length %d, expected %d (compressed trials)",
                          res.size(), n_comp);
             }

             // 4) Copy into out_ (compressed trajectory)
             out_.assign(n_comp, 0.0);
             for (int j = 0; j < n_comp; ++j) {
               out_[j] = res[j];
             }

             mark_run_complete();
           }
};

// For sequential kernels: currently same as BaseKernel -- just included to allow for other types (e.g. Bayesian ideal observer, autoregressive) in the future
struct SequentialKernel : BaseKernel {
  virtual ~SequentialKernel() {}
};

// All 1D delta kernels have scalar q and 1D pes_
struct DeltaKernel : SequentialKernel {
protected:
  double q_ = NA_REAL;             // latest value
  double pe_ = NA_REAL;            // latest PE
  std::vector<double> pes_;        // PE per trial
  const int* q_reset_ = nullptr;   // <-- ADD: null = no reset

public:
  virtual ~DeltaKernel() {}

  void set_kernel_args(const KernelArgs& args) override {
    q_reset_ = args.q_reset;
  }

  // const std::vector<double>& get_pes() const {
  //   return pes_;
  // }

  bool has_output_stream(int code) const override {
    return (code >= 1 && code <= 2);
  }

  KernelOutput get_output_stream(int code) const override {
    const int n_full = static_cast<int>(out_.size());

    if (code == 1) {
      return KernelOutput{ out_.data(), n_full, 1 };
    }

    if (code == 2) {
      stream_buf_[1].resize(n_full);
      if (!has_expand_idx_) {
        if ((int)pes_.size() != n_full)
          Rcpp::stop("DeltaKernel: pes_ length mismatch");
        for (int i = 0; i < n_full; ++i) stream_buf_[1][i] = pes_[i];
      } else {
        const auto& idx = expand_idx_;
        for (int i = 0; i < n_full; ++i) stream_buf_[1][i] = pes_[idx[i] - 1];
      }
      return KernelOutput{ stream_buf_[1].data(), n_full, 1 };
    }

    Rcpp::stop("DeltaKernel::get_output_stream: unsupported code %d (1=Q,2=PE)", code);
  }


  // Rcpp::NumericVector get_output_stream(int code) const override {
  //   using namespace Rcpp;
  //
  //   const int n_full = static_cast<int>(out_.size());
  //
  //   if (code == 1) {
  //     // main trajectory: already full-length
  //     return wrap(out_);
  //   }
  //
  //   if (code == 2) {
  //     NumericVector res(n_full);
  //
  //     if (!has_expand_idx_) {
  //       // no 'at': one-to-one
  //       if ((int)pes_.size() != n_full)
  //         stop("DeltaKernel: pes_ length mismatch");
  //       for (int i = 0; i < n_full; ++i) res[i] = pes_[i];
  //     } else {
  //       // with 'at': expand from compressed index
  //       const auto& idx = expand_idx_;
  //       if ((int)idx.size() != n_full)
  //         stop("DeltaKernel: expand_idx length mismatch");
  //       for (int i = 0; i < n_full; ++i) {
  //         int k = idx[i] - 1;  // compressed index
  //         res[i] = pes_[k];
  //       }
  //     }
  //     return res;
  //   }
  //
  //   stop("DeltaKernel::get_output_stream: unsupported code %d (1=Q,2=PE)", code);
  // }

  std::string output_stream_name(int code) const override {
    if (code == 1) return "Qvalue";
    if (code == 2) return "PE";
    throw std::runtime_error("DeltaKernel::output_stream_name: unsupported code");
  }
};


// ---- Individual kernels ----
// ---- Non-sequential kernels ----

struct LinIncrKernel : BaseKernel {
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             int n_comp = comp_idx.size();
             out_.assign(n_comp, 0);    // compressed output

             for (int j = 0; j < n_comp; ++j) {
               int r = comp_idx[j];
               double x = covariate(r,0);
               out_[j] = x;
               // if (!is_nan(x)) {
               //   out_[j] = x;  // compressed index
               // }
             }

             mark_run_complete();
           }
};


struct LinDecrKernel : BaseKernel {
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             int n_comp = comp_idx.size();
             out_.assign(n_comp, 0);

             // double last = NA_REAL;
             for (int j = 0; j < n_comp; ++j) {
               int r = comp_idx[j];
               double x = covariate(r,0);
               out_[j] = -x;
               // if (!is_nan(x)) {
               //   out_[j] = -x;
               // }
               // out_[j] = last;
             }

             mark_run_complete();
           }
};

struct ExpDecrKernel : BaseKernel {
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             if (kernel_pars.cols.size() != 1) {
               Rcpp::stop("ExpDecrKernel expects 1 parameter columns, got %d",
                          (int)kernel_pars.cols.size());
             }

             int n_comp = comp_idx.size();
             out_.assign(n_comp, 0);

             const double* lambda_col = kernel_pars.cols[0];
             // double last = NA_REAL;
             for (int j = 0; j < n_comp; ++j) {
               int r = comp_idx[j];
               double x = covariate(r,0);
               // if (!is_nan(x)) {
                 double lambda = lambda_col[r];
                 out_[j] = std::exp(-lambda * x);
               // }
               // out_[j] = last;
             }

             mark_run_complete();
           }
};

struct ExpIncrKernel : BaseKernel {
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             if (kernel_pars.cols.size() != 1) {
               Rcpp::stop("ExpIncrKernel expects 1 parameter columns, got %d",
                          (int)kernel_pars.cols.size());
             }

             int n_comp = comp_idx.size();
             out_.assign(n_comp, 0);

             const double* lambda_col = kernel_pars.cols[0];
             // double last = NA_REAL;
             for (int j = 0; j < n_comp; ++j) {
               int r = comp_idx[j];
               double x = covariate(r,0);
               // if (!is_nan(x)) {
                 double lambda = lambda_col[r];
                 out_[j] = 1.0 - std::exp(-lambda * x);
               // }
               // out_[j] = last;
             }

             mark_run_complete();
           }
};

struct PowDecrKernel : BaseKernel {
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             if (kernel_pars.cols.size() != 1) {
               Rcpp::stop("PowDecrKernel expects 1 parameter columns, got %d",
                          (int)kernel_pars.cols.size());
             }

             int n_comp = comp_idx.size();
             out_.assign(n_comp, 0);

             const double* alpha_col = kernel_pars.cols[0];
             // double last = NA_REAL;
             for (int j = 0; j < n_comp; ++j) {
               int r = comp_idx[j];
               double x = covariate(r,0);
               // if (!is_nan(x)) {
                 double alpha = alpha_col[r];
                 out_[j] = std::pow(1.0 + x, -alpha);
               // }
               // out_[j] = last;
             }

             mark_run_complete();
           }
};

struct PowIncrKernel : BaseKernel {
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             if (kernel_pars.cols.size() != 1) {
               Rcpp::stop("PowIncrKernel expects 1 parameter columns, got %d",
                          (int)kernel_pars.cols.size());
             }

             int n_comp = comp_idx.size();
             out_.assign(n_comp, 0);

             const double* alpha_col = kernel_pars.cols[0];
             // double last = NA_REAL;
             for (int j = 0; j < n_comp; ++j) {
               int r = comp_idx[j];
               double x = covariate(r,0);
               // if (!is_nan(x)) {
                 double alpha = alpha_col[r];
                 out_[j] = 1.0 - std::pow(1.0 + x, -alpha);
               // }
               // out_[j] = last;
             }

             mark_run_complete();
           }
};

struct Poly2Kernel : BaseKernel {
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 2) {
               Rcpp::stop("Poly2Kernel expects 2 parameter columns, got %d",
                          (int)kernel_pars.cols.size());
             }

             int n_comp = comp_idx.size();
             out_.assign(n_comp, 0);

             const double* a1_col = kernel_pars.cols[0];
             const double* a2_col = kernel_pars.cols[1];

             // double last = NA_REAL;
             for (int j = 0; j < n_comp; ++j) {
               int r = comp_idx[j];
               double x = covariate(r,0);
               // if (!is_nan(x)) {
                 double a1 = a1_col[r];
                 double a2 = a2_col[r];
                 double x2 = x * x;
                 out_[j] = a1 * x + a2 * x2;
               // }
               // out_[j] = last;
             }

             mark_run_complete();
           }
};

struct Poly3Kernel : BaseKernel {
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 3) {
               Rcpp::stop("Poly3Kernel expects 3 parameter columns, got %d",
                          (int)kernel_pars.cols.size());
             }

             int n_comp = comp_idx.size();
             out_.assign(n_comp, 0);

             const double* a1_col = kernel_pars.cols[0];
             const double* a2_col = kernel_pars.cols[1];
             const double* a3_col = kernel_pars.cols[2];

             // double last = NA_REAL;
             for (int j = 0; j < n_comp; ++j) {
               int r = comp_idx[j];
               double x = covariate(r,0);
               // if (!is_nan(x)) {
                 double a1 = a1_col[r];
                 double a2 = a2_col[r];
                 double a3 = a3_col[r];
                 double x2 = x * x;
                 double x3 = x2 * x;
                 out_[j] = a1 * x + a2 * x2 + a3 * x3;
               // }
               // out_[j] = last;
             }

             mark_run_complete();
           }
};

struct Poly4Kernel : BaseKernel {
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 4) {
               Rcpp::stop("Poly4Kernel expects 4 parameter columns, got %d",
                          (int)kernel_pars.cols.size());
             }

             int n_comp = comp_idx.size();
             out_.assign(n_comp, 0);

             const double* a1_col = kernel_pars.cols[0];
             const double* a2_col = kernel_pars.cols[1];
             const double* a3_col = kernel_pars.cols[2];
             const double* a4_col = kernel_pars.cols[3];

             // double last = NA_REAL;
             for (int j = 0; j < n_comp; ++j) {
               int r = comp_idx[j];
               double x = covariate(r,0);
               // if (!is_nan(x)) {
                 double a1 = a1_col[r];
                 double a2 = a2_col[r];
                 double a3 = a3_col[r];
                 double a4 = a4_col[r];

                 double x2 = x * x;
                 double x3 = x2 * x;
                 double x4 = x2 * x2;
                 out_[j] = a1 * x + a2 * x2 + a3 * x3 + a4 * x4;
               // }
               // out_[j] = last;
             }

             mark_run_complete();
           }
};


// Sequential kernels
struct SimpleDelta : DeltaKernel {
  SimpleDelta() {}

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 2) {
               Rcpp::stop("SimpleDelta expects 2 parameter columns, got %d",
                          (int)kernel_pars.cols.size());
             }

             const int n_comp = static_cast<int>(comp_idx.size());
             if (n_comp <= 0) {
               out_.clear();
               pes_.clear();
               mark_run_complete();
               return;
             }

             // slightly faster -- no-op size check, no need to write anything
             out_.resize(n_comp);
             pes_.resize(n_comp);
             pes_[n_comp - 1] = NA_REAL;

             const double* q0_col    = kernel_pars.cols[0];
             const double* alpha_col = kernel_pars.cols[1];
             const double* cov_ptr   = covariate.colptr(0);

             int row0 = comp_idx[0];
             q_       = q0_col[row0];
             out_[0]  = q_;

             for (int j = 0; j < n_comp - 1; ++j) {
               int r    = comp_idx[j];
               // --- RESET (before PE) ---
               if (q_reset_ && q_reset_[r]) {
                 q_ = q0_col[r];
                 out_[j] = q_;           // overwrite with reset value
               }

               double x = cov_ptr[r];

               if (!is_nan(x)) {
                 double alpha = alpha_col[r];
                 double pe    = x - q_;
                 pes_[j]      = pe;
                 q_          += alpha * pe;
               } else {
                 pes_[j] = NA_REAL;
               }
               out_[j + 1] = q_;
             }

             mark_run_complete();
           }
};

// DeltaExpdecrKernel — plain delta with an exp-DECREASING learning rate.
// Like SimpleDelta but alpha_eff(t) = Phi(alpha_base + alpha_w * exp(-d_alpha_ed * block_trial[r])),
// where alpha_base is on the probit scale and alpha_w, d_alpha_ed are exp-transformed (>= 0);
// block_trial = within-block 0-based exposure count (kernel_args$block_trial_column).
// d_alpha_ed -> 0 or alpha_w -> 0 nests constant-rate delta. supports_grouping, so a multi-column
// cov_names expands to one independent delta per feature (e.g. the sum-Q urgency channel).
// Parameters: q0, alpha_base, alpha_w, d_alpha_ed.
// Ported from EMC2-oo_refactor_simd_winall/src/kernels.h:819 (covariate is a Mat, not NumericMatrix).
struct DeltaExpdecrKernel : DeltaKernel {
private:
  const double* block_trial_ = nullptr;

  double normal_cdf(double x) const {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
  }

  double learning_rate(int r,
                       const double* alpha_base_col,
                       const double* alpha_w_col,
                       const double* d_alpha_ed_col) const {
    double t = block_trial_ ? block_trial_[r] : 0.0;
    if (is_nan(t) || t < 0.0) t = 0.0;
    double d_alpha_ed = d_alpha_ed_col[r];
    if (is_nan(d_alpha_ed) || d_alpha_ed < 0.0) d_alpha_ed = 0.0;
    double alpha_base = alpha_base_col[r];
    if (is_nan(alpha_base)) alpha_base = 0.0;
    double alpha_w = alpha_w_col[r];
    if (is_nan(alpha_w) || alpha_w < 0.0) alpha_w = 0.0;
    return normal_cdf(alpha_base + alpha_w * std::exp(-d_alpha_ed * t));
  }

public:
  void set_kernel_args(const KernelArgs& args) override {
    q_reset_     = args.q_reset;
    block_trial_ = args.block_trial;
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 4) {
               Rcpp::stop("DeltaExpdecrKernel expects 4 parameter columns (q0, alpha_base, alpha_w, d_alpha_ed), got %d",
                          (int)kernel_pars.cols.size());
             }

             const int n_comp = static_cast<int>(comp_idx.size());
             if (n_comp <= 0) {
               out_.clear();
               pes_.clear();
               mark_run_complete();
               return;
             }

             if (!block_trial_) {
               Rcpp::stop("DeltaExpdecrKernel requires kernel_args$block_trial_column");
             }

             out_.resize(n_comp);
             pes_.resize(n_comp);
             pes_[n_comp - 1] = NA_REAL;

             const double* q0_col         = kernel_pars.cols[0];
             const double* alpha_base_col = kernel_pars.cols[1];
             const double* alpha_w_col    = kernel_pars.cols[2];
             const double* d_alpha_ed_col = kernel_pars.cols[3];
             const double* cov_ptr        = covariate.colptr(0);

             int row0 = comp_idx[0];
             q_       = q0_col[row0];
             out_[0]  = q_;

             for (int j = 0; j < n_comp - 1; ++j) {
               int r = comp_idx[j];
               if (q_reset_ && q_reset_[r]) {
                 q_ = q0_col[r];
                 out_[j] = q_;
               }

               double x = cov_ptr[r];
               if (!is_nan(x)) {
                 double alpha = learning_rate(r, alpha_base_col, alpha_w_col, d_alpha_ed_col);
                 double pe    = x - q_;
                 pes_[j]      = pe;
                 q_          += alpha * pe;
               } else {
                 pes_[j] = NA_REAL;
               }
               out_[j + 1] = q_;
             }

             mark_run_complete();
           }
};

struct Delta2LR : DeltaKernel {
  Delta2LR() {}

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 3) {
               Rcpp::stop("Delta2LR expects 3 parameter columns, got %d",
                          (int)kernel_pars.cols.size());
             }

             int n_comp = comp_idx.size();
             out_.assign(n_comp, NA_REAL);
             pes_.assign(n_comp, NA_REAL);

             const double* q0_col       = kernel_pars.cols[0];
             const double* alphaPos_col = kernel_pars.cols[1];
             const double* alphaNeg_col = kernel_pars.cols[2];

             int row0 = comp_idx[0];
             out_[0] = q_ = q0_col[row0];

             double pe = NA_REAL;

             for (int j = 0; j < n_comp - 1; ++j) {
               int r = comp_idx[j];

               // --- RESET (before PE) ---
               if (q_reset_ && q_reset_[r]) {
                 q_ = q0_col[r];
                 out_[j] = q_;           // overwrite with reset value
               }

               double x = covariate(r,0);
               if (!is_nan(x)) {
                 double alphaPos = alphaPos_col[r];
                 double alphaNeg = alphaNeg_col[r];
                 pe = x - q_;
                 double alpha = (pe > 0.0) ? alphaPos : alphaNeg;
                 q_ += alpha * pe;
               } else {
                 pe = NA_REAL;
               }

               pes_[j] = pe;           // compressed index
               out_[j + 1] = q_;
             }

             mark_run_complete();
           }
};

// 2D PE kernel: separate from DeltaKernel
struct Delta2Kernel : SequentialKernel {
  double qFast_ = NA_REAL;
  double qSlow_ = NA_REAL;
  double q_     = NA_REAL;
  const int* q_reset_ = nullptr;

  void set_kernel_args(const KernelArgs& args) override {
    q_reset_ = args.q_reset;
  }

  // [compressed trial][0 = fast PE, 1 = slow PE]
  std::vector<double> pes_fast_;
  std::vector<double> pes_slow_;
  std::vector<double> q_fast_;
  std::vector<double> q_slow_;

  // One dedicated transpose buffer per secondary stream code.
  // Indexed as: secondary_buf_[code - 2], i.e.:
  //   code 2 (Qfast)   -> secondary_buf_[0]
  //   code 3 (Qslow)   -> secondary_buf_[1]
  //   code 4 (PEfast)  -> secondary_buf_[2]
  //   code 5 (PEslow)  -> secondary_buf_[3]
  mutable std::vector<double> secondary_buf_[4];

  Delta2Kernel() {}

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 4) {
               Rcpp::stop("Delta2Kernel expects 4 parameter columns, got %d",
                          (int)kernel_pars.cols.size());
             }

             int n_comp = comp_idx.size();
             out_.assign(n_comp, NA_REAL);
             q_fast_.assign(n_comp, NA_REAL);
             q_slow_.assign(n_comp, NA_REAL);
             pes_fast_.assign(n_comp, NA_REAL);
             pes_slow_.assign(n_comp, NA_REAL);

             const double* q0_col        = kernel_pars.cols[0];
             const double* alphaFast_col = kernel_pars.cols[1];
             const double* propSlow_col  = kernel_pars.cols[2];
             const double* dSwitch_col   = kernel_pars.cols[3];

             int row0 = comp_idx[0];
             out_[0] = qFast_ = qSlow_ = q_ = q0_col[row0];

             for (int j = 0; j < n_comp - 1; ++j) {
               int r = comp_idx[j];

               // --- RESET (before PE): both trackers reset to q0 ---
               if (q_reset_ && q_reset_[r]) {
                 qFast_ = qSlow_ = q_ = q0_col[r];
                 out_[j] = q_;           // overwrite with reset value
               }

               double x = covariate(r,0);
               double peFast = NA_REAL;
               double peSlow = NA_REAL;

               if (!is_nan(x)) {
                 double alphaFast = alphaFast_col[r];
                 double propSlow  = propSlow_col[r];
                 double dSwitch   = dSwitch_col[r];
                 double alphaSlow = propSlow * alphaFast;

                 peFast = x - qFast_;
                 peSlow = x - qSlow_;

                 qFast_ += alphaFast * peFast;
                 qSlow_ += alphaSlow * peSlow;

                 double diff = std::abs(qFast_ - qSlow_);
                 q_ = (diff > dSwitch) ? qFast_ : qSlow_;
               }
               q_fast_[j+1] = qFast_;  // compressed index
               q_slow_[j+1] = qSlow_;

               pes_fast_[j] = peFast;  // compressed index
               pes_slow_[j] = peSlow;
               out_[j + 1] = q_;
             }

             mark_run_complete();
           }

  bool has_output_stream(int code) const override {
    return (code >= 1 && code <= 5);
  }

  KernelOutput get_output_stream(int code) const override {
    const int n_full = static_cast<int>(out_.size());

    if (code == 1) {
      return KernelOutput{ out_.data(), n_full, 1 };
    }

    const std::vector<double>* src = nullptr;
    if      (code == 2) src = &q_fast_;
    else if (code == 3) src = &q_slow_;
    else if (code == 4) src = &pes_fast_;
    else if (code == 5) src = &pes_slow_;
    else Rcpp::stop("Delta2Kernel::get_output_stream: unsupported code %d "
                      "(1=Q, 2=Qfast, 3=Qslow, 4=PEfast, 5=PEslow)", code);

    std::vector<double>& buf = secondary_buf_[code - 2];
    buf.resize(n_full);

    if (!has_expand_idx_) {
      if ((int)src->size() != n_full)
        Rcpp::stop("Delta2Kernel::get_output_stream: source length mismatch");
      for (int i = 0; i < n_full; ++i) buf[i] = (*src)[i];
    } else {
      const auto& idx = expand_idx_;
      for (int i = 0; i < n_full; ++i) buf[i] = (*src)[idx[i] - 1];
    }

    return KernelOutput{ buf.data(), n_full, 1 };
  }

  std::string output_stream_name(int code) const override {
    if (code == 1) return "Qvalue";
    if (code == 2) return "Qfast";
    if (code == 3) return "Qslow";
    if (code == 4) return "PEfast";
    if (code == 5) return "PEslow";
    throw std::runtime_error("Delta2Kernel::output_stream_name: unsupported code");
  }
};


struct RescorlaWagnerKernel : SequentialKernel {
private:
  // Row-major internal storage: index as [r * n_covs_ + col]
  int n_covs_ = 0;
  std::vector<double> q_mat_;   // [n_comp * n_covs_]: Q-value per trial per covariate
  std::vector<double> pe_mat_;  // [n_comp * n_covs_]: compound PE for active covariates, NA otherwise

  const int* q_reset_ = nullptr;

public:
  void set_kernel_args(const KernelArgs& args) override {
    q_reset_ = args.q_reset;
  }

  void reset() override {
    BaseKernel::reset();
    q_mat_.clear();
    pe_mat_.clear();
    n_covs_ = 0;
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             if (kernel_pars.cols.size() != 2) {
               Rcpp::stop("RescorlaWagnerKernel expects 2 parameter columns (q0, alpha), got %d",
                          (int)kernel_pars.cols.size());
             }

             const int n_comp = static_cast<int>(comp_idx.size());
             n_covs_          = covariate.ncol;

             if (n_comp == 0 || n_covs_ == 0) {
               q_mat_.clear();
               pe_mat_.clear();
               mark_run_complete();
               return;
             }

             const double* q0_col    = kernel_pars.cols[0];
             const double* alpha_col = kernel_pars.cols[1];

             q_mat_.assign(n_comp * n_covs_, NA_REAL);
             pe_mat_.assign(n_comp * n_covs_, NA_REAL);

             // Initialise: all covariates start at q0 of first trial
             int row0       = comp_idx[0];
             double q0_init = q0_col[row0];

             std::vector<double> q_cur(n_covs_, q0_init);

             // Write Q-values entering trial 0
             for (int c = 0; c < n_covs_; ++c) {
               q_mat_[0 * n_covs_ + c] = q_cur[c];
             }

             for (int j = 0; j < n_comp - 1; ++j) {
               int r = comp_idx[j];

               // Reset before PE: overwrite q_cur and the already-written q_mat_[j]
               if (q_reset_ && q_reset_[r]) {
                 double q0_r = q0_col[r];
                 for (int c = 0; c < n_covs_; ++c) {
                   q_cur[c]                  = q0_r;
                   q_mat_[j * n_covs_ + c]  = q0_r;  // overwrite entering Q for this trial
                 }
               }

               // Identify active covariates and accumulate compound Q
               double reward    = NA_REAL;
               double q_active  = 0.0;
               bool   any_active = false;

               for (int c = 0; c < n_covs_; ++c) {
                 double x = covariate(r, c);
                 if (!is_nan(x)) {
                   reward     = x;   // reward is the same across all active columns
                   q_active  += q_cur[c];
                   any_active = true;
                 }
               }

               if (any_active && !is_nan(reward)) {
                 double alpha       = alpha_col[r];
                 double compound_pe = reward - q_active;

                 for (int c = 0; c < n_covs_; ++c) {
                   double x = covariate(r, c);
                   if (!is_nan(x)) {
                     pe_mat_[j * n_covs_ + c] = compound_pe;
                     q_cur[c] += alpha * compound_pe;
                   }
                   // inactive: pe_mat_ stays NA, q_cur[c] unchanged
                 }
               }

               // Write Q-values entering trial j+1
               for (int c = 0; c < n_covs_; ++c) {
                 q_mat_[(j + 1) * n_covs_ + c] = q_cur[c];
               }
               // Rprintf("n_covs_=%d n_comp=%d\n", n_covs_, n_comp);
             }

             // pe_mat_ for the last trial stays NA (no outcome consumed yet),
             // mirroring SimpleDelta's pes_[n_comp - 1] = NA_REAL

             mark_run_complete();
           }

  bool has_output_stream(int code) const override {
    return (code == 1 || code == 2);
  }

  // Stream 1: Q-matrix (n_rows x n_covs_), column-major
  // Stream 2: PE-matrix (n_rows x n_covs_), column-major
  KernelOutput get_output_stream(int code) const override {
    if (code != 1 && code != 2) {
      Rcpp::stop("RescorlaWagnerKernel::get_output_stream: unsupported code %d "
                   "(1=Qmatrix, 2=PEmatrix)", code);
    }

    const std::vector<double>& src = (code == 1) ? q_mat_ : pe_mat_;
    const int n_comp  = static_cast<int>(src.size()) / n_covs_;
    std::vector<double>& buf = stream_buf_[code - 1];

    if (has_expand_idx_) {
      // Expand compact [n_comp x n_covs_] to full [n_trials x n_covs_],
      // then transpose to column-major for R.
      const int n_full = static_cast<int>(expand_idx_.size());
      buf.resize(n_full * n_covs_);

      for (int c = 0; c < n_covs_; ++c) {
        for (int i = 0; i < n_full; ++i) {
          int comp_row = expand_idx_[i] - 1;  // 1-based -> 0-based
          buf[c * n_full + i] = src[comp_row * n_covs_ + c];
        }
      }

      return KernelOutput{ buf.data(), n_full, n_covs_ };

    } else {
      // No expand: just transpose row-major [n_comp x n_covs_] to column-major
      buf.resize(n_comp * n_covs_);

      for (int c = 0; c < n_covs_; ++c) {
        for (int r = 0; r < n_comp; ++r) {
          buf[c * n_comp + r] = src[r * n_covs_ + c];
        }
      }

      return KernelOutput{ buf.data(), n_comp, n_covs_ };
    }
  }

  std::string output_stream_name(int code) const override {
    if (code == 1) return "Qmatrix";
    if (code == 2) return "PEmatrix";
    throw std::runtime_error("RescorlaWagnerKernel::output_stream_name: unsupported code");
  }
};


// // 2kernel adjusted
// struct Delta2Kernel2 : Delta2Kernel {
//   double qFast_ = NA_REAL;
//   double qSlow_ = NA_REAL;
//   double q_     = NA_REAL;
//
//   Delta2Kernel2() {}
//
//   void run(const KernelParsView& kernel_pars,
//            const Rcpp::NumericMatrix& covariate,
//            const std::vector<int>& comp_idx) override {
//              if (kernel_pars.cols.size() != 4) {
//                Rcpp::stop("Delta2Kernel expects 4 parameter columns, got %d",
//                           (int)kernel_pars.cols.size());
//              }
//
//              int n_comp = comp_idx.size();
//              out_.assign(n_comp, NA_REAL);
//              q_fast_.assign(n_comp, NA_REAL);
//              q_slow_.assign(n_comp, NA_REAL);
//              pes_fast_.assign(n_comp, NA_REAL);
//              pes_slow_.assign(n_comp, NA_REAL);
//
//              const double* q0_col        = kernel_pars.cols[0];
//              const double* alphaFast_col = kernel_pars.cols[1];
//              const double* propSlow_col  = kernel_pars.cols[2];
//              const double* dSwitch_col   = kernel_pars.cols[3];
//
//              int row0 = comp_idx[0];
//              out_[0] = qFast_ = qSlow_ = q_ = q0_col[row0];
//              int current_kernel = 0; // 0 = fast, 1 = slow
//
//              for (int j = 0; j < n_comp - 1; ++j) {
//                int r = comp_idx[j];
//                double x = covariate(r,0);
//                double peFast = NA_REAL;
//                double peSlow = NA_REAL;
//
//                if (!ISNAN(x)) {
//                  double alphaFast = alphaFast_col[r];
//                  double propSlow  = propSlow_col[r];
//                  double dSwitch   = dSwitch_col[r];
//                  double alphaSlow = propSlow * alphaFast;
//
//                  peFast = x - qFast_;
//                  peSlow = x - qSlow_;
//
//                  qFast_ += alphaFast * peFast;
//                  qSlow_ += alphaSlow * peSlow;
//
//                  double diff = std::abs(qFast_ - qSlow_);
//                  if(diff > dSwitch) {
//                    current_kernel = 0; // fast kernel
//                    q_ = qFast_;
//                  } else {
//                    if(current_kernel == 0) {
//                      // was in fast mode, now moving to slow mode. Override Q-value of slow
//                      qSlow_ = qFast_;
//                    }
//                    current_kernel = 1;
//                    q_ = qSlow_;
//                  }
//                  // q_ = (diff > dSwitch) ? qFast_ : qSlow_;
//                }
//
//                q_fast_[j+1] = qFast_;  // compressed index
//                q_slow_[j+1] = qSlow_;
//
//                pes_fast_[j] = peFast;  // compressed index
//                pes_slow_[j] = peSlow;
//                out_[j + 1] = q_;
//              }
//
//              mark_run_complete();
//            }
// };

// =============================================================================
// DBMBaseKernel
// Streams: 1 = prediction mean, 2 = prediction mode, 3 = surprise (bits)
// =============================================================================

struct DBMBaseKernel : BaseKernel {
protected:
  std::vector<double> pred_mean_;
  std::vector<double> pred_mode_;
  mutable std::vector<double> surprise_;          // computed lazily
  std::vector<double> comp_obs_;                  // compressed observations, stored during run()
  mutable bool surprise_computed_ = false;

  void store_obs(const double* cov_ptr, const std::vector<int>& comp_idx) {
    const int n_comp = static_cast<int>(comp_idx.size());
    comp_obs_.resize(n_comp);
    for (int j = 0; j < n_comp; ++j)
      comp_obs_[j] = cov_ptr[comp_idx[j]];
  }

  void ensure_surprise() const {
    if (surprise_computed_) return;
    const int n_comp = static_cast<int>(pred_mean_.size());
    surprise_.resize(n_comp, std::numeric_limits<double>::quiet_NaN());
    for (int j = 0; j < n_comp; ++j)
      if (!is_nan(comp_obs_[j]))
        surprise_[j] = shannon_surprise(pred_mean_[j], comp_obs_[j]);
    surprise_computed_ = true;
  }

public:
  void reset() override {
    BaseKernel::reset();
    pred_mean_.clear();
    pred_mode_.clear();
    surprise_.clear();
    comp_obs_.clear();
    surprise_computed_ = false;
  }

  bool has_output_stream(int code) const override {
    return (code >= 1 && code <= 3);
  }

  KernelOutput get_output_stream(int code) const override {
    if (code == 3) ensure_surprise(); // compute surprise the moment it is requested, not before
    const std::vector<double>* src = nullptr;
    if      (code == 1) src = &pred_mean_;
    else if (code == 2) src = &pred_mode_;
    else if (code == 3) src = &surprise_;
    else Rcpp::stop("DBMBaseKernel::get_output_stream: unsupported code %d "
                      "(1=mean, 2=mode, 3=surprise)", code);

    if (has_expand_idx_) {
      const int n_full = static_cast<int>(expand_idx_.size());
      stream_buf_[0].resize(n_full);
      for (int i = 0; i < n_full; ++i)
        stream_buf_[0][i] = (*src)[expand_idx_[i] - 1];
      return KernelOutput{ stream_buf_[0].data(), n_full, 1 };
    }
    return KernelOutput{ src->data(), static_cast<int>(src->size()), 1 };
  }

  std::string output_stream_name(int code) const override {
    if (code == 1) return "mean";
    if (code == 2) return "mode";
    if (code == 3) return "surprise";
    throw std::runtime_error("DBMBaseKernel::output_stream_name: unsupported code");
  }

protected:
  void compute_surprise(const double* cov_ptr,
                        const std::vector<int>& comp_idx) {
    const int n_comp = static_cast<int>(comp_idx.size());
    surprise_.resize(n_comp, std::numeric_limits<double>::quiet_NaN());
    for (int j = 0; j < n_comp; ++j) {
      const double obs = cov_ptr[comp_idx[j]];
      if (!is_nan(obs))
        surprise_[j] = shannon_surprise(pred_mean_[j], obs);
    }
  }

  // keep out_ in sync with pred_mean_ so BaseKernel::do_expand works if called
  void sync_out_to_mean() { out_ = pred_mean_; }
};

// =============================================================================
// BetaBinomialKernel  —  basic (no memory constraint)
// Parameters: a0, b0
// =============================================================================

struct BetaBinomialKernel : DBMBaseKernel {
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             if (kernel_pars.cols.size() != 2)
               Rcpp::stop("BetaBinomialKernel expects 2 parameter columns (a0, b0), got %d",
                          (int)kernel_pars.cols.size());

             const int     n_comp  = static_cast<int>(comp_idx.size());
             const double* a0_col  = kernel_pars.cols[0];
             const double* b0_col  = kernel_pars.cols[1];
             const double* cov_ptr = covariate.colptr(0);

             pred_mean_.resize(n_comp);
             pred_mode_.resize(n_comp);

             double n_hit = 0.0, n_trial = 0.0;

             for (int j = 0; j < n_comp; ++j) {
               const int    r   = comp_idx[j];
               const double a_t = a0_col[r] + n_hit;
               const double b_t = b0_col[r] + (n_trial - n_hit);

               pred_mean_[j] = beta_mean(a_t, b_t);
               pred_mode_[j] = beta_mode(a_t, b_t);

               const double x = cov_ptr[r];
               if (!is_nan(x)) { n_hit += x; n_trial += 1.0; }
             }

             store_obs(cov_ptr, comp_idx);
             sync_out_to_mean();
             mark_run_complete();
           }
};

// =============================================================================
// BetaBinomialDecayKernel  —  exponential decay on accumulated counts
// Parameters: a0, b0, decay
// =============================================================================

struct BetaBinomialDecayKernel : DBMBaseKernel {
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             if (kernel_pars.cols.size() != 3)
               Rcpp::stop("BetaBinomialDecayKernel expects 3 parameter columns "
                            "(a0, b0, decay), got %d",
                            (int)kernel_pars.cols.size());

             const int     n_comp    = static_cast<int>(comp_idx.size());
             const double* a0_col    = kernel_pars.cols[0];
             const double* b0_col    = kernel_pars.cols[1];
             const double* decay_col = kernel_pars.cols[2];
             const double* cov_ptr   = covariate.colptr(0);

             pred_mean_.resize(n_comp);
             pred_mode_.resize(n_comp);

             double n_hit = 0.0, n_trial = 0.0;

             for (int j = 0; j < n_comp; ++j) {
               const int    r   = comp_idx[j];
               const double a_t = a0_col[r] + n_hit;
               const double b_t = b0_col[r] + (n_trial - n_hit);

               pred_mean_[j] = beta_mean(a_t, b_t);
               pred_mode_[j] = beta_mode(a_t, b_t);

               const double df = std::exp(-1.0 / decay_col[r]);
               const double x  = cov_ptr[r];
               if (!is_nan(x)) {
                 n_hit   = df * (n_hit + x);
                 n_trial = df * (n_trial + 1.0);
               } else {
                 // still decay without observation: passage of time erodes memory
                 n_hit   = df * n_hit;
                 n_trial = df * n_trial;
               }
             }

             store_obs(cov_ptr, comp_idx);
             sync_out_to_mean();
             mark_run_complete();
           }
};

// =============================================================================
// BetaBinomialWindowKernel  —  fixed sliding window
// Parameters: a0, b0, window
// =============================================================================

struct BetaBinomialWindowKernel : DBMBaseKernel {
private:
  struct Event { double obs; int idx; };

public:
  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             if (kernel_pars.cols.size() != 3)
               Rcpp::stop("BetaBinomialWindowKernel expects 3 parameter columns "
                            "(a0, b0, window), got %d",
                            (int)kernel_pars.cols.size());

             const int     n_comp     = static_cast<int>(comp_idx.size());
             const double* a0_col     = kernel_pars.cols[0];
             const double* b0_col     = kernel_pars.cols[1];
             const double* window_col = kernel_pars.cols[2];
             const double* cov_ptr    = covariate.colptr(0);

             pred_mean_.resize(n_comp);
             pred_mode_.resize(n_comp);

             double n_hit = 0.0, n_trial = 0.0;
             std::deque<Event> buf;

             for (int j = 0; j < n_comp; ++j) {
               const int r = comp_idx[j];
               const int w = static_cast<int>(window_col[r]);

               // prune observations outside the window
               while (!buf.empty() && (r - buf.front().idx) > w) {
                 n_hit   -= buf.front().obs;
                 n_trial -= 1.0;
                 buf.pop_front();
               }

               const double a_t = a0_col[r] + n_hit;
               const double b_t = b0_col[r] + (n_trial - n_hit);

               pred_mean_[j] = beta_mean(a_t, b_t);
               pred_mode_[j] = beta_mode(a_t, b_t);

               const double x = cov_ptr[r];
               if (!is_nan(x)) {
                 buf.push_back({x, r});
                 n_hit   += x;
                 n_trial += 1.0;
               }
             }

             store_obs(cov_ptr, comp_idx);
             sync_out_to_mean();
             mark_run_complete();
           }
};

// =============================================================================
// DBMKernel  —  Dynamic Belief Model
// Yu & Cohen (2008), NeurIPS; Ide et al. (2013), JoN
// Parameters: cp, mu0, s0
// kernel_args: grid_res (default 100)
// =============================================================================

struct DBMKernel : DBMBaseKernel {
private:
  int grid_res_ = 100;

public:
  void set_kernel_args(const KernelArgs& args) override {
    if (args.grid_res > 0) grid_res_ = args.grid_res;
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             if (kernel_pars.cols.size() != 3)
               Rcpp::stop("DBMKernel expects 3 parameter columns (cp, mu0, s0), got %d",
                          (int)kernel_pars.cols.size());

             const int     n_comp  = static_cast<int>(comp_idx.size());
             const double* cp_col  = kernel_pars.cols[0];
             const double* mu0_col = kernel_pars.cols[1];
             const double* s0_col  = kernel_pars.cols[2];
             const double* cov_ptr = covariate.colptr(0);

             pred_mean_.resize(n_comp);
             pred_mode_.resize(n_comp);

             const double cp_eps = 1e-10;
             const int    gs     = grid_res_ + 1;

             std::vector<double> prob_grid(gs), x_like(gs), y_like(gs);
             for (int i = 0; i < gs; ++i) {
               prob_grid[i] = static_cast<double>(i) / (gs - 1);
               x_like[i]   = prob_grid[i];
               y_like[i]   = 1.0 - prob_grid[i];
             }

             std::vector<double> DBM_prior(gs), DBM_pred(gs), DBM_post(gs);

             for (int j = 0; j < n_comp; ++j) {
               const int    r   = comp_idx[j];
               const double cp  = cp_col[r];
               const double mu0 = mu0_col[r];
               const double s0  = s0_col[r];
               const double a   = mu0 * s0;
               const double b   = (1.0 - mu0) * s0;
               const double x   = cov_ptr[r];

               // degenerate: cp ≈ 1 — beliefs driven by fixed prior only
               if ((1.0 - cp) < cp_eps) {
                 pred_mean_[j] = beta_mean(a, b);
                 pred_mode_[j] = beta_mode(a, b);
                 for (int i = 0; i < gs; ++i)
                   DBM_post[i] = dbeta_val(prob_grid[i], a, b);
                 normalise_inplace(DBM_post);
                 continue;
               }

               // compute discretised Beta prior
               for (int i = 0; i < gs; ++i)
                 DBM_prior[i] = dbeta_val(prob_grid[i], a, b);
               normalise_inplace(DBM_prior);

               // predictive distribution
               if (j == 0 || cp < cp_eps) {
                 // first trial or cp ≈ 0: prior is the predictive
                 DBM_pred = DBM_prior;
               } else {
                 const double mix_old = 1.0 - cp;
                 const double mix_new = cp;
                 for (int i = 0; i < gs; ++i)
                   DBM_pred[i] = mix_old * DBM_post[i] + mix_new * DBM_prior[i];
                 normalise_inplace(DBM_pred);
               }

               pred_mean_[j] = mean_discrete(prob_grid, DBM_pred);
               pred_mode_[j] = mode_discrete(prob_grid, DBM_pred);

               // posterior update
               if (is_nan(x)) {
                 DBM_post = DBM_pred;   // no observation: push predictive forward
               } else {
                 const std::vector<double>& like = (x == 1.0) ? x_like : y_like;
                 for (int i = 0; i < gs; ++i)
                   DBM_post[i] = DBM_pred[i] * like[i];
                 normalise_inplace(DBM_post);
               }
             }

             store_obs(cov_ptr, comp_idx);
             sync_out_to_mean();
             mark_run_complete();
           }
};

// =============================================================================
// TPMKernel  —  Transition Probability Model
// Meyniel et al. (2016), PLOS CB
// Parameters: cp, a0, b0
// kernel_args: grid_res (default 100)
// =============================================================================

struct TPMKernel : DBMBaseKernel {
private:
  int grid_res_ = 100;

  struct TPMGrid {
    int resol = 0, n_combi = 0;
    std::vector<double> p_XX, p_XY;
    std::vector<double> like_XX, like_XY, like_YX, like_YY;
    std::vector<double> mean_p;
  };

  TPMGrid build_grid(int grid_res) const {
    const int resol   = grid_res + 1;
    const int n_combi = resol * resol;

    std::vector<double> grid(resol);
    for (int i = 0; i < resol; ++i)
      grid[i] = static_cast<double>(i) / (resol - 1);

    TPMGrid g;
    g.resol = resol; g.n_combi = n_combi;
    g.p_XX.resize(n_combi);    g.p_XY.resize(n_combi);
    g.like_XX.resize(n_combi); g.like_XY.resize(n_combi);
    g.like_YX.resize(n_combi); g.like_YY.resize(n_combi);
    g.mean_p.resize(n_combi);

    int idx = 0;
    for (int i0 = 0; i0 < resol; ++i0) {
      const double pXY = grid[i0];
      for (int i1 = 0; i1 < resol; ++i1) {
        const double pXX   = grid[i1];
        g.p_XX[idx]    = pXX;
        g.p_XY[idx]    = pXY;
        g.like_XX[idx] = pXX;
        g.like_XY[idx] = pXY;
        g.like_YX[idx] = 1.0 - pXX;
        g.like_YY[idx] = 1.0 - pXY;
        g.mean_p[idx]  = 0.5 * (pXX + pXY);
        ++idx;
      }
    }
    return g;
  }

public:
  void set_kernel_args(const KernelArgs& args) override {
    if (args.grid_res > 0) grid_res_ = args.grid_res;
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {

             if (kernel_pars.cols.size() != 3)
               Rcpp::stop("TPMKernel expects 3 parameter columns (cp, a0, b0), got %d",
                          (int)kernel_pars.cols.size());

             const int     n_comp  = static_cast<int>(comp_idx.size());
             const double* cp_col  = kernel_pars.cols[0];
             const double* a0_col  = kernel_pars.cols[1];
             const double* b0_col  = kernel_pars.cols[2];
             const double* cov_ptr = covariate.colptr(0);

             pred_mean_.resize(n_comp);
             pred_mode_.resize(n_comp);

             const double cp_eps  = 1e-10;
             const TPMGrid grid   = build_grid(grid_res_);
             const int     nc     = grid.n_combi;
             const double  inv_nm1 = 1.0 / (nc - 1.0);

             std::vector<double> TPM_post(nc), TPM_pred(nc), TPM_update(nc);

             // initialise posterior with Beta prior from first trial
             {
               const int r0 = comp_idx[0];
               for (int k = 0; k < nc; ++k)
                 TPM_post[k] = dbeta_val(grid.p_XX[k], a0_col[r0], b0_col[r0])
                 * dbeta_val(grid.p_XY[k], a0_col[r0], b0_col[r0]);
               normalise_inplace(TPM_post);
             }

             for (int j = 0; j < n_comp; ++j) {
               const int    r       = comp_idx[j];
               const double cp      = cp_col[r];
               const double x       = cov_ptr[r];
               const bool   curr_na = is_nan(x);
               const bool   prev_na = (j == 0) || is_nan(cov_ptr[comp_idx[j - 1]]);
               const int    curr    = curr_na ? -1 : static_cast<int>(x);
               const int    prev    = (j == 0 || prev_na) ? -1
               : static_cast<int>(cov_ptr[comp_idx[j - 1]]);

               // degenerate: cp ≈ 1
               if ((1.0 - cp) < cp_eps) {
                 pred_mean_[j] = beta_mean(a0_col[r], b0_col[r]);
                 pred_mode_[j] = pred_mean_[j];
                 continue;
               }

               // degenerate: cp ≈ 0 — no volatility, read directly from posterior
               if (cp < cp_eps) {
                 pred_mean_[j] = prev_na
                 ? mean_discrete(grid.mean_p, TPM_post)
                   : (prev == 1 ? mean_discrete(grid.p_XX, TPM_post)
                        : mean_discrete(grid.p_XY, TPM_post));
                 pred_mode_[j] = pred_mean_[j];
               } else {
                 // full TPM update
                 const double sum_post = std::accumulate(
                   TPM_post.begin(), TPM_post.end(), 0.0);
                 const double mix_old  = 1.0 - cp;
                 const double mix_new  = cp;

                 for (int k = 0; k < nc; ++k)
                   TPM_pred[k] = mix_old * TPM_post[k]
                 + mix_new * (sum_post - TPM_post[k]) * inv_nm1;
                 normalise_inplace(TPM_pred);

                 pred_mean_[j] = prev_na
                 ? mean_discrete(grid.mean_p, TPM_pred)
                   : (prev == 1 ? mean_discrete(grid.p_XX, TPM_pred)
                        : mean_discrete(grid.p_XY, TPM_pred));
                 pred_mode_[j] = pred_mean_[j];

                 if (curr_na || prev_na) {
                   TPM_post = TPM_pred;
                 } else {
                   const std::vector<double>* lp =
                     (prev == 0) ? (curr == 0 ? &grid.like_YY : &grid.like_XY)
                     : (curr == 0 ? &grid.like_YX : &grid.like_XX);
                   for (int k = 0; k < nc; ++k)
                     TPM_update[k] = mix_old * (*lp)[k] * TPM_post[k]
                   + mix_new * (*lp)[k]
                   * (sum_post - TPM_post[k]) * inv_nm1;
                   normalise_inplace(TPM_update);
                   std::swap(TPM_post, TPM_update);
                 }
               }
             }

             store_obs(cov_ptr, comp_idx);
             sync_out_to_mean();
             mark_run_complete();
           }
};


// =============================================================================
// DeltaSatlinkGammaCardKernel — CARDINALITY-saturation satlink read-out (delta)
// Ported from EMC2-oo_refactor_simd_winall/src/kernels.h:1341-1547 (only change:
// the run() covariate argument is a Mat, not an Rcpp::NumericMatrix).
//
// Variadic: takes all n_elem covariate columns at once. Learns one Q per elemental
// feature with an independent delta update, forms the directional difference
//   D = (q[feat_one_1] + q[feat_one_2]) - (q[feat_two_1] + q[feat_two_2])
// from per-trial 1-based feature-index columns (kernel_args), applies the saturating
// link g = sat * tanh(D^gamma / sat) with sat chosen by VALUATION CARDINALITY
// (a present second feature on either option -> sat_double; else sat_single; the
// configural channel passes no *_2 columns so it is always sat_single), and emits the
// compressed AllocShape row [g, -g, 0, ...] over n_elem columns (read out per
// accumulator by make_alloc_selector_map). q0 arrives already transformed (pnorm).
// Parameters: q0, alpha, sat_single, sat_double, gamma.
// =============================================================================
struct DeltaSatlinkGammaCardKernel : SequentialKernel {
private:
  int n_elem_ = 0;
  int n_covs_ = 0;

  const int* q_reset_ = nullptr;
  const int* feat_one_1_idx_ = nullptr;
  const int* feat_one_2_idx_ = nullptr;
  const int* feat_two_1_idx_ = nullptr;
  const int* feat_two_2_idx_ = nullptr;
  int arg_n_elem_ = -1;

  // Compressed [n_comp x n_elem_] allocation shape, row-major: [g, -g, 0, ...]
  std::vector<double> q_mat_;

  static double sat_link(double d, double theta, double gamma) {
    double dpow = d;
    if (!is_nan(gamma) && gamma > 0.0 && gamma != 1.0) {
      const double ad = std::fabs(d);
      dpow = (d < 0.0 ? -1.0 : 1.0) * std::pow(ad, gamma);
    }
    if (is_nan(theta) || theta <= 0.0) return dpow;
    if (theta > 1e6) return dpow;     // saturation off; avoid 0*inf overflow
    return theta * std::tanh(dpow / theta);
  }

  int required_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) {
      Rcpp::stop("DeltaSatlinkGammaCardKernel requires kernel_args$%s", arg_name);
    }
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) {
      Rcpp::stop("DeltaSatlinkGammaCardKernel requires kernel_args$%s[%d] to be a positive 1-based elemental index",
                 arg_name, row + 1);
    }
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("DeltaSatlinkGammaCardKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  int optional_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) return -1;
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) return -1;
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("DeltaSatlinkGammaCardKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  void emit_row(int j, int r,
                const double* sat_single_col, const double* sat_double_col,
                const double* gamma_col,
                const std::vector<double>& q) {
    const int i11 = required_feat_index(feat_one_1_idx_, r, "feat_one_1_idx_column");
    const int i12 = optional_feat_index(feat_one_2_idx_, r, "feat_one_2_idx_column");
    const int i21 = required_feat_index(feat_two_1_idx_, r, "feat_two_1_idx_column");
    const int i22 = optional_feat_index(feat_two_2_idx_, r, "feat_two_2_idx_column");

    const double q11 = q[i11];
    const double q12 = (i12 >= 0) ? q[i12] : 0.0;
    const double q21 = q[i21];
    const double q22 = (i22 >= 0) ? q[i22] : 0.0;

    const double d = (q11 + q12) - (q21 + q22);

    // Cardinality of THIS valuation: a present second feature on EITHER option marks a
    // two-item (Double) read-out -> sat_double; otherwise one item -> sat_single. The
    // configural channel passes no feat_*_2 columns, so it is always one-item -> sat_single.
    const bool two_item = (i12 >= 0) || (i22 >= 0);
    const double sat = two_item ? sat_double_col[r] : sat_single_col[r];
    const double g = sat_link(d, sat, gamma_col[r]);

    double* row_out = &q_mat_[j * n_elem_];
    row_out[0] =  g;   // shape_one: contribution when lR == UV_One
    row_out[1] = -g;   // shape_two: contribution when lR == UV_Two
    for (int c = 2; c < n_elem_; ++c) row_out[c] = 0.0;
  }

public:
  void set_kernel_args(const KernelArgs& args) override {
    q_reset_ = args.q_reset;
    feat_one_1_idx_ = args.feat_one_1_idx;
    feat_one_2_idx_ = args.feat_one_2_idx;
    feat_two_1_idx_ = args.feat_two_1_idx;
    feat_two_2_idx_ = args.feat_two_2_idx;
    arg_n_elem_ = args.n_elem;
  }

  void reset() override {
    BaseKernel::reset();
    q_mat_.clear();
    n_elem_ = 0;
    n_covs_ = 0;
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 5) {
               Rcpp::stop("DeltaSatlinkGammaCardKernel expects 5 parameter columns (q0, alpha, sat_single, sat_double, gamma), got %d",
                          (int)kernel_pars.cols.size());
             }

             const int n_comp = static_cast<int>(comp_idx.size());
             n_covs_ = covariate.ncol;
             n_elem_ = arg_n_elem_;

             if (n_comp == 0 || n_covs_ == 0) {
               q_mat_.clear();
               mark_run_complete();
               return;
             }

             if (n_elem_ < 2) {
               Rcpp::stop("DeltaSatlinkGammaCardKernel requires kernel_args$n_elem >= 2 to form a directional difference; got n_elem=%d",
                          n_elem_);
             }
             if (n_covs_ != n_elem_) {
               Rcpp::stop("DeltaSatlinkGammaCardKernel requires the number of covariate columns (%d) to equal kernel_args$n_elem (%d)",
                          n_covs_, n_elem_);
             }
             if (!feat_one_1_idx_ || !feat_two_1_idx_) {
               Rcpp::stop("DeltaSatlinkGammaCardKernel requires kernel_args$feat_one_1_idx_column and kernel_args$feat_two_1_idx_column");
             }

             const double* q0_col         = kernel_pars.cols[0];
             const double* alpha_col      = kernel_pars.cols[1];
             const double* sat_single_col = kernel_pars.cols[2];
             const double* sat_double_col = kernel_pars.cols[3];
             const double* gamma_col      = kernel_pars.cols[4];

             const int row0 = comp_idx[0];
             std::vector<double> q(n_elem_, q0_col[row0]);
             q_mat_.assign(n_comp * n_elem_, 0.0);

             emit_row(0, row0, sat_single_col, sat_double_col, gamma_col, q);

             for (int j = 0; j < n_comp - 1; ++j) {
               const int r = comp_idx[j];

               if (q_reset_ && q_reset_[r] == 1) {
                 std::fill(q.begin(), q.end(), q0_col[r]);
                 emit_row(j, r, sat_single_col, sat_double_col, gamma_col, q);
               }

               const double alpha = alpha_col[r];
               for (int e = 0; e < n_elem_; ++e) {
                 const double x = covariate(r, e);
                 if (!is_nan(x)) {
                   q[e] += alpha * (x - q[e]);
                 }
               }

               const int next_r = comp_idx[j + 1];
               emit_row(j + 1, next_r, sat_single_col, sat_double_col, gamma_col, q);
             }

             mark_run_complete();
           }

  KernelOutput get_output_stream(int code) const override {
    if (code != 1) {
      Rcpp::stop("DeltaSatlinkGammaCardKernel::get_output_stream: unsupported code %d (1=AllocShape)", code);
    }

    if (n_elem_ <= 0) return KernelOutput{ q_mat_.data(), 0, 0 };

    const int n_comp = static_cast<int>(q_mat_.size()) / n_elem_;
    std::vector<double>& buf = stream_buf_[0];

    if (has_expand_idx_) {
      const int n_full = static_cast<int>(expand_idx_.size());
      buf.resize(n_full * n_elem_);

      for (int c = 0; c < n_elem_; ++c) {
        for (int i = 0; i < n_full; ++i) {
          int comp_row = expand_idx_[i] - 1;
          buf[c * n_full + i] = q_mat_[comp_row * n_elem_ + c];
        }
      }

      return KernelOutput{ buf.data(), n_full, n_elem_ };
    }

    buf.resize(n_comp * n_elem_);

    for (int c = 0; c < n_elem_; ++c) {
      for (int r = 0; r < n_comp; ++r) {
        buf[c * n_comp + r] = q_mat_[r * n_elem_ + c];
      }
    }

    return KernelOutput{ buf.data(), n_comp, n_elem_ };
  }

  std::string output_stream_name(int code) const override {
    if (code == 1) return "AllocShape";
    throw std::runtime_error("DeltaSatlinkGammaCardKernel::output_stream_name: unsupported code");
  }
};


// =============================================================================
// DeltaSatlinkGammaCardExpdecrKernel — CARDINALITY-saturation satlink read-out
// (delta) with an exp-DECREASING learning rate. = DeltaSatlinkGammaCardKernel +
// the DeltaExpdecr schedule. Identical cardsat geometry/read-out (per-feature
// independent delta, directional D, cardinality-selected sat, AllocShape g) EXCEPT
// the per-trial learning rate is
//   alpha_eff(t) = Phi(alpha_base + alpha_w * exp(-d_alpha_ed * block_trial[r]))
// (alpha_base on probit scale; alpha_w, d_alpha_ed exp-transformed >= 0; block_trial
// = within-block 0-based exposure count). d_alpha_ed -> 0 or alpha_w -> 0 nests the
// constant-rate DeltaSatlinkGammaCardKernel.
// Parameters: q0, alpha_base, alpha_w, d_alpha_ed, sat_single, sat_double, gamma.
// =============================================================================
struct DeltaSatlinkGammaCardExpdecrKernel : SequentialKernel {
private:
  int n_elem_ = 0;
  int n_covs_ = 0;

  const int* q_reset_ = nullptr;
  const int* feat_one_1_idx_ = nullptr;
  const int* feat_one_2_idx_ = nullptr;
  const int* feat_two_1_idx_ = nullptr;
  const int* feat_two_2_idx_ = nullptr;
  const double* block_trial_ = nullptr;
  int arg_n_elem_ = -1;

  // Compressed [n_comp x n_elem_] allocation shape, row-major: [g, -g, 0, ...]
  std::vector<double> q_mat_;

  // Identical to DeltaSatlinkGammaCardKernel::sat_link.
  static double sat_link(double d, double theta, double gamma) {
    double dpow = d;
    if (!is_nan(gamma) && gamma > 0.0 && gamma != 1.0) {
      const double ad = std::fabs(d);
      dpow = (d < 0.0 ? -1.0 : 1.0) * std::pow(ad, gamma);
    }
    if (is_nan(theta) || theta <= 0.0) return dpow;
    if (theta > 1e6) return dpow;     // saturation off; avoid 0*inf overflow
    return theta * std::tanh(dpow / theta);
  }

  double normal_cdf(double x) const {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
  }

  // alpha_eff(t) = Phi(alpha_base + alpha_w * exp(-d_alpha_ed * block_trial[r])).
  double learning_rate(int r,
                       const double* alpha_base_col,
                       const double* alpha_w_col,
                       const double* d_alpha_ed_col) const {
    double t = block_trial_ ? block_trial_[r] : 0.0;
    if (is_nan(t) || t < 0.0) t = 0.0;
    double d_alpha_ed = d_alpha_ed_col[r];
    if (is_nan(d_alpha_ed) || d_alpha_ed < 0.0) d_alpha_ed = 0.0;
    double alpha_base = alpha_base_col[r];
    if (is_nan(alpha_base)) alpha_base = 0.0;
    double alpha_w = alpha_w_col[r];
    if (is_nan(alpha_w) || alpha_w < 0.0) alpha_w = 0.0;
    return normal_cdf(alpha_base + alpha_w * std::exp(-d_alpha_ed * t));
  }

  int required_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) {
      Rcpp::stop("DeltaSatlinkGammaCardExpdecrKernel requires kernel_args$%s", arg_name);
    }
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) {
      Rcpp::stop("DeltaSatlinkGammaCardExpdecrKernel requires kernel_args$%s[%d] to be a positive 1-based elemental index",
                 arg_name, row + 1);
    }
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("DeltaSatlinkGammaCardExpdecrKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  int optional_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) return -1;
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) return -1;
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("DeltaSatlinkGammaCardExpdecrKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  void emit_row(int j, int r,
                const double* sat_single_col, const double* sat_double_col,
                const double* gamma_col,
                const std::vector<double>& q) {
    const int i11 = required_feat_index(feat_one_1_idx_, r, "feat_one_1_idx_column");
    const int i12 = optional_feat_index(feat_one_2_idx_, r, "feat_one_2_idx_column");
    const int i21 = required_feat_index(feat_two_1_idx_, r, "feat_two_1_idx_column");
    const int i22 = optional_feat_index(feat_two_2_idx_, r, "feat_two_2_idx_column");

    const double q11 = q[i11];
    const double q12 = (i12 >= 0) ? q[i12] : 0.0;
    const double q21 = q[i21];
    const double q22 = (i22 >= 0) ? q[i22] : 0.0;

    const double d = (q11 + q12) - (q21 + q22);

    const bool two_item = (i12 >= 0) || (i22 >= 0);
    const double sat = two_item ? sat_double_col[r] : sat_single_col[r];
    const double g = sat_link(d, sat, gamma_col[r]);

    double* row_out = &q_mat_[j * n_elem_];
    row_out[0] =  g;
    row_out[1] = -g;
    for (int c = 2; c < n_elem_; ++c) row_out[c] = 0.0;
  }

public:
  void set_kernel_args(const KernelArgs& args) override {
    q_reset_ = args.q_reset;
    feat_one_1_idx_ = args.feat_one_1_idx;
    feat_one_2_idx_ = args.feat_one_2_idx;
    feat_two_1_idx_ = args.feat_two_1_idx;
    feat_two_2_idx_ = args.feat_two_2_idx;
    block_trial_ = args.block_trial;
    arg_n_elem_ = args.n_elem;
  }

  void reset() override {
    BaseKernel::reset();
    q_mat_.clear();
    n_elem_ = 0;
    n_covs_ = 0;
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 7) {
               Rcpp::stop("DeltaSatlinkGammaCardExpdecrKernel expects 7 parameter columns (q0, alpha_base, alpha_w, d_alpha_ed, sat_single, sat_double, gamma), got %d",
                          (int)kernel_pars.cols.size());
             }

             const int n_comp = static_cast<int>(comp_idx.size());
             n_covs_ = covariate.ncol;
             n_elem_ = arg_n_elem_;

             if (n_comp == 0 || n_covs_ == 0) {
               q_mat_.clear();
               mark_run_complete();
               return;
             }

             if (n_elem_ < 2) {
               Rcpp::stop("DeltaSatlinkGammaCardExpdecrKernel requires kernel_args$n_elem >= 2 to form a directional difference; got n_elem=%d",
                          n_elem_);
             }
             if (n_covs_ != n_elem_) {
               Rcpp::stop("DeltaSatlinkGammaCardExpdecrKernel requires the number of covariate columns (%d) to equal kernel_args$n_elem (%d)",
                          n_covs_, n_elem_);
             }
             if (!feat_one_1_idx_ || !feat_two_1_idx_) {
               Rcpp::stop("DeltaSatlinkGammaCardExpdecrKernel requires kernel_args$feat_one_1_idx_column and kernel_args$feat_two_1_idx_column");
             }
             if (!block_trial_) {
               Rcpp::stop("DeltaSatlinkGammaCardExpdecrKernel requires kernel_args$block_trial_column");
             }

             const double* q0_col         = kernel_pars.cols[0];
             const double* alpha_base_col = kernel_pars.cols[1];
             const double* alpha_w_col    = kernel_pars.cols[2];
             const double* d_alpha_ed_col = kernel_pars.cols[3];
             const double* sat_single_col = kernel_pars.cols[4];
             const double* sat_double_col = kernel_pars.cols[5];
             const double* gamma_col      = kernel_pars.cols[6];

             const int row0 = comp_idx[0];
             std::vector<double> q(n_elem_, q0_col[row0]);
             q_mat_.assign(n_comp * n_elem_, 0.0);

             emit_row(0, row0, sat_single_col, sat_double_col, gamma_col, q);

             for (int j = 0; j < n_comp - 1; ++j) {
               const int r = comp_idx[j];

               if (q_reset_ && q_reset_[r] == 1) {
                 std::fill(q.begin(), q.end(), q0_col[r]);
                 emit_row(j, r, sat_single_col, sat_double_col, gamma_col, q);
               }

               const double alpha = learning_rate(r, alpha_base_col, alpha_w_col, d_alpha_ed_col);
               for (int e = 0; e < n_elem_; ++e) {
                 const double x = covariate(r, e);
                 if (!is_nan(x)) {
                   q[e] += alpha * (x - q[e]);
                 }
               }

               const int next_r = comp_idx[j + 1];
               emit_row(j + 1, next_r, sat_single_col, sat_double_col, gamma_col, q);
             }

             mark_run_complete();
           }

  KernelOutput get_output_stream(int code) const override {
    if (code != 1) {
      Rcpp::stop("DeltaSatlinkGammaCardExpdecrKernel::get_output_stream: unsupported code %d (1=AllocShape)", code);
    }

    if (n_elem_ <= 0) return KernelOutput{ q_mat_.data(), 0, 0 };

    const int n_comp = static_cast<int>(q_mat_.size()) / n_elem_;
    std::vector<double>& buf = stream_buf_[0];

    if (has_expand_idx_) {
      const int n_full = static_cast<int>(expand_idx_.size());
      buf.resize(n_full * n_elem_);

      for (int c = 0; c < n_elem_; ++c) {
        for (int i = 0; i < n_full; ++i) {
          int comp_row = expand_idx_[i] - 1;
          buf[c * n_full + i] = q_mat_[comp_row * n_elem_ + c];
        }
      }

      return KernelOutput{ buf.data(), n_full, n_elem_ };
    }

    buf.resize(n_comp * n_elem_);

    for (int c = 0; c < n_elem_; ++c) {
      for (int r = 0; r < n_comp; ++r) {
        buf[c * n_comp + r] = q_mat_[r * n_elem_ + c];
      }
    }

    return KernelOutput{ buf.data(), n_comp, n_elem_ };
  }

  std::string output_stream_name(int code) const override {
    if (code == 1) return "AllocShape";
    throw std::runtime_error("DeltaSatlinkGammaCardExpdecrKernel::output_stream_name: unsupported code");
  }
};


// =============================================================================
// DeltaSatlinkGammaCardPhKernel — CARDINALITY-saturation satlink read-out with an
// INDEPENDENT-prediction-error PEARCE-HALL dynamic learning rate.
// = DeltaSatlinkGammaCardKernel (per-feature independent delta + cardsat read-out)
// but the FIXED rate alpha is replaced by a per-cue PEARCE-HALL associability that
// tracks THAT cue's OWN surprise. Unlike RwSatlinkGammaCardPhKernel (one summed
// compound PE shared across the chosen option's cues), here each active cue e uses
// its OWN prediction error pe_e = reward_e - q[e] for BOTH updates:
//   q[e]     += assoc[e] * pe_e               // value update at the cue's rate
//   assoc[e]  = eta*|pe_e| + (1-eta)*assoc[e] // associability relaxes toward |pe_e|
// assoc IS the learning rate, clamped to [0,1]; alpha0 = initial associability,
// eta = mixing weight. Inactive cues (NA covariate) carry q AND assoc forward. Same
// cardsat geometry/read-out as DeltaSatlinkGammaCard: NO 3rd cue; a configural
// channel passing only feat_*_1 is always sat_single. eta -> 0 nests the constant
// independent-delta rate alpha0. q0 = plogis, alpha0/eta = pnorm in (0,1),
// sat_*/gamma = exp. Parameters: q0, alpha0, eta, sat_single, sat_double, gamma.
// =============================================================================
struct DeltaSatlinkGammaCardPhKernel : SequentialKernel {
private:
  int n_elem_ = 0;
  int n_covs_ = 0;

  const int* q_reset_ = nullptr;
  const int* feat_one_1_idx_ = nullptr;
  const int* feat_one_2_idx_ = nullptr;
  const int* feat_two_1_idx_ = nullptr;
  const int* feat_two_2_idx_ = nullptr;
  int arg_n_elem_ = -1;

  // Compressed [n_comp x n_elem_] allocation shape, row-major: [g, -g, 0, ...]
  std::vector<double> q_mat_;

  // Identical to DeltaSatlinkGammaCardKernel::sat_link.
  static double sat_link(double d, double theta, double gamma) {
    double dpow = d;
    if (!is_nan(gamma) && gamma > 0.0 && gamma != 1.0) {
      const double ad = std::fabs(d);
      dpow = (d < 0.0 ? -1.0 : 1.0) * std::pow(ad, gamma);
    }
    if (is_nan(theta) || theta <= 0.0) return dpow;
    if (theta > 1e6) return dpow;     // saturation off; avoid 0*inf overflow
    return theta * std::tanh(dpow / theta);
  }

  int required_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) {
      Rcpp::stop("DeltaSatlinkGammaCardPhKernel requires kernel_args$%s", arg_name);
    }
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) {
      Rcpp::stop("DeltaSatlinkGammaCardPhKernel requires kernel_args$%s[%d] to be a positive 1-based elemental index",
                 arg_name, row + 1);
    }
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("DeltaSatlinkGammaCardPhKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  int optional_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) return -1;
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) return -1;
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("DeltaSatlinkGammaCardPhKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  void emit_row(int j, int r,
                const double* sat_single_col, const double* sat_double_col,
                const double* gamma_col,
                const std::vector<double>& q) {
    const int i11 = required_feat_index(feat_one_1_idx_, r, "feat_one_1_idx_column");
    const int i12 = optional_feat_index(feat_one_2_idx_, r, "feat_one_2_idx_column");
    const int i21 = required_feat_index(feat_two_1_idx_, r, "feat_two_1_idx_column");
    const int i22 = optional_feat_index(feat_two_2_idx_, r, "feat_two_2_idx_column");

    const double q11 = q[i11];
    const double q12 = (i12 >= 0) ? q[i12] : 0.0;
    const double q21 = q[i21];
    const double q22 = (i22 >= 0) ? q[i22] : 0.0;

    const double d = (q11 + q12) - (q21 + q22);

    const bool two_item = (i12 >= 0) || (i22 >= 0);
    const double sat = two_item ? sat_double_col[r] : sat_single_col[r];
    const double g = sat_link(d, sat, gamma_col[r]);

    double* row_out = &q_mat_[j * n_elem_];
    row_out[0] =  g;
    row_out[1] = -g;
    for (int c = 2; c < n_elem_; ++c) row_out[c] = 0.0;
  }

public:
  void set_kernel_args(const KernelArgs& args) override {
    q_reset_ = args.q_reset;
    feat_one_1_idx_ = args.feat_one_1_idx;
    feat_one_2_idx_ = args.feat_one_2_idx;
    feat_two_1_idx_ = args.feat_two_1_idx;
    feat_two_2_idx_ = args.feat_two_2_idx;
    arg_n_elem_ = args.n_elem;
  }

  void reset() override {
    BaseKernel::reset();
    q_mat_.clear();
    n_elem_ = 0;
    n_covs_ = 0;
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 6) {
               Rcpp::stop("DeltaSatlinkGammaCardPhKernel expects 6 parameter columns (q0, alpha0, eta, sat_single, sat_double, gamma), got %d",
                          (int)kernel_pars.cols.size());
             }

             const int n_comp = static_cast<int>(comp_idx.size());
             n_covs_ = covariate.ncol;
             n_elem_ = arg_n_elem_;

             if (n_comp == 0 || n_covs_ == 0) {
               q_mat_.clear();
               mark_run_complete();
               return;
             }

             if (n_elem_ < 2) {
               Rcpp::stop("DeltaSatlinkGammaCardPhKernel requires kernel_args$n_elem >= 2 to form a directional difference; got n_elem=%d",
                          n_elem_);
             }
             if (n_covs_ != n_elem_) {
               Rcpp::stop("DeltaSatlinkGammaCardPhKernel requires the number of covariate columns (%d) to equal kernel_args$n_elem (%d)",
                          n_covs_, n_elem_);
             }
             if (!feat_one_1_idx_ || !feat_two_1_idx_) {
               Rcpp::stop("DeltaSatlinkGammaCardPhKernel requires kernel_args$feat_one_1_idx_column and kernel_args$feat_two_1_idx_column");
             }

             const double* q0_col         = kernel_pars.cols[0];
             const double* alpha0_col     = kernel_pars.cols[1];
             const double* eta_col        = kernel_pars.cols[2];
             const double* sat_single_col = kernel_pars.cols[3];
             const double* sat_double_col = kernel_pars.cols[4];
             const double* gamma_col      = kernel_pars.cols[5];

             const int row0 = comp_idx[0];
             std::vector<double> q(n_elem_, q0_col[row0]);
             std::vector<double> assoc(n_elem_, alpha0_col[row0]);  // per-cue associability (PH rate)
             q_mat_.assign(n_comp * n_elem_, 0.0);

             emit_row(0, row0, sat_single_col, sat_double_col, gamma_col, q);

             for (int j = 0; j < n_comp - 1; ++j) {
               const int r = comp_idx[j];

               if (q_reset_ && q_reset_[r] == 1) {
                 std::fill(q.begin(), q.end(), q0_col[r]);
                 std::fill(assoc.begin(), assoc.end(), alpha0_col[r]);  // reset associability too
                 emit_row(j, r, sat_single_col, sat_double_col, gamma_col, q);
               }

               // INDEPENDENT per-cue prediction error: each active (non-NA cov) cue e uses
               // its OWN pe_e = reward_e - q[e] for the value update AND the Pearce-Hall
               // associability relaxation. No shared/summed compound PE (that is the RW
               // variant). assoc[e] IS the learning rate, so it is clamped to [0,1].
               double eta = eta_col[r];
               if (is_nan(eta)) eta = 0.0;
               if (eta < 0.0) eta = 0.0;
               if (eta > 1.0) eta = 1.0;
               for (int e = 0; e < n_elem_; ++e) {
                 const double x = covariate(r, e);
                 if (!is_nan(x)) {
                   const double pe = x - q[e];
                   q[e] += assoc[e] * pe;                  // rate = current associability
                   double a = eta * std::fabs(pe) + (1.0 - eta) * assoc[e];
                   if (a < 0.0) a = 0.0;
                   if (a > 1.0) a = 1.0;                   // clamp: assoc IS the LR -> keep in [0,1]
                   assoc[e] = a;
                 }
               }

               const int next_r = comp_idx[j + 1];
               emit_row(j + 1, next_r, sat_single_col, sat_double_col, gamma_col, q);
             }

             mark_run_complete();
           }

  KernelOutput get_output_stream(int code) const override {
    if (code != 1) {
      Rcpp::stop("DeltaSatlinkGammaCardPhKernel::get_output_stream: unsupported code %d (1=AllocShape)", code);
    }

    if (n_elem_ <= 0) return KernelOutput{ q_mat_.data(), 0, 0 };

    const int n_comp = static_cast<int>(q_mat_.size()) / n_elem_;
    std::vector<double>& buf = stream_buf_[0];

    if (has_expand_idx_) {
      const int n_full = static_cast<int>(expand_idx_.size());
      buf.resize(n_full * n_elem_);

      for (int c = 0; c < n_elem_; ++c) {
        for (int i = 0; i < n_full; ++i) {
          int comp_row = expand_idx_[i] - 1;
          buf[c * n_full + i] = q_mat_[comp_row * n_elem_ + c];
        }
      }

      return KernelOutput{ buf.data(), n_full, n_elem_ };
    }

    buf.resize(n_comp * n_elem_);

    for (int c = 0; c < n_elem_; ++c) {
      for (int r = 0; r < n_comp; ++r) {
        buf[c * n_comp + r] = q_mat_[r * n_elem_ + c];
      }
    }

    return KernelOutput{ buf.data(), n_comp, n_elem_ };
  }

  std::string output_stream_name(int code) const override {
    if (code == 1) return "AllocShape";
    throw std::runtime_error("DeltaSatlinkGammaCardPhKernel::output_stream_name: unsupported code");
  }
};


// =============================================================================
// RwSatlinkGammaCardKernel — CARDINALITY-saturation satlink read-out with
// RESCORLA-WAGNER (shared/summed prediction-error) learning. = the cardsat
// read-out + the RW summed-error update: instead of an INDEPENDENT delta per
// feature, the prediction is the SUM of the chosen option's active cue Q's and
// ONE shared PE = (reward - sum) updates them all (cue competition). Hybrid use:
// pass the configural compound as an OPTIONAL 3rd cue per option (feat_one_3 /
// feat_two_3) so the shared PE spans {feat_1, feat_2, compound} and the read-out
// difference is D = (q11+q12+q13) - (q21+q22+q23). Cardinality (sat_single vs
// sat_double) is decided by the ELEMENTAL second feature only (the compound rides
// the same sat). Parameters: q0, alpha, sat_single, sat_double, gamma.
// =============================================================================
struct RwSatlinkGammaCardKernel : SequentialKernel {
private:
  int n_elem_ = 0;
  int n_covs_ = 0;

  const int* q_reset_ = nullptr;
  const int* feat_one_1_idx_ = nullptr;
  const int* feat_one_2_idx_ = nullptr;
  const int* feat_one_3_idx_ = nullptr;
  const int* feat_two_1_idx_ = nullptr;
  const int* feat_two_2_idx_ = nullptr;
  const int* feat_two_3_idx_ = nullptr;
  int arg_n_elem_ = -1;

  // Compressed [n_comp x n_elem_] allocation shape, row-major: [g, -g, 0, ...]
  std::vector<double> q_mat_;

  static double sat_link(double d, double theta, double gamma) {
    double dpow = d;
    if (!is_nan(gamma) && gamma > 0.0 && gamma != 1.0) {
      const double ad = std::fabs(d);
      dpow = (d < 0.0 ? -1.0 : 1.0) * std::pow(ad, gamma);
    }
    if (is_nan(theta) || theta <= 0.0) return dpow;
    if (theta > 1e6) return dpow;     // saturation off; avoid 0*inf overflow
    return theta * std::tanh(dpow / theta);
  }

  int required_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) {
      Rcpp::stop("RwSatlinkGammaCardKernel requires kernel_args$%s", arg_name);
    }
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) {
      Rcpp::stop("RwSatlinkGammaCardKernel requires kernel_args$%s[%d] to be a positive 1-based index",
                 arg_name, row + 1);
    }
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("RwSatlinkGammaCardKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  int optional_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) return -1;
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) return -1;
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("RwSatlinkGammaCardKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  void emit_row(int j, int r,
                const double* sat_single_col, const double* sat_double_col,
                const double* gamma_col,
                const std::vector<double>& q) {
    const int i11 = required_feat_index(feat_one_1_idx_, r, "feat_one_1_idx_column");
    const int i12 = optional_feat_index(feat_one_2_idx_, r, "feat_one_2_idx_column");
    const int i13 = optional_feat_index(feat_one_3_idx_, r, "feat_one_3_idx_column");
    const int i21 = required_feat_index(feat_two_1_idx_, r, "feat_two_1_idx_column");
    const int i22 = optional_feat_index(feat_two_2_idx_, r, "feat_two_2_idx_column");
    const int i23 = optional_feat_index(feat_two_3_idx_, r, "feat_two_3_idx_column");

    const double q11 = q[i11];
    const double q12 = (i12 >= 0) ? q[i12] : 0.0;
    const double q13 = (i13 >= 0) ? q[i13] : 0.0;
    const double q21 = q[i21];
    const double q22 = (i22 >= 0) ? q[i22] : 0.0;
    const double q23 = (i23 >= 0) ? q[i23] : 0.0;

    const double d = (q11 + q12 + q13) - (q21 + q22 + q23);

    // Cardinality from the ELEMENTAL second feature only (a present feat_*_2 marks a
    // two-item Double read-out). The configural compound (feat_*_3) rides the same sat.
    const bool two_item = (i12 >= 0) || (i22 >= 0);
    const double sat = two_item ? sat_double_col[r] : sat_single_col[r];
    const double g = sat_link(d, sat, gamma_col[r]);

    double* row_out = &q_mat_[j * n_elem_];
    row_out[0] =  g;   // contribution when lR == UV_One
    row_out[1] = -g;   // contribution when lR == UV_Two
    for (int c = 2; c < n_elem_; ++c) row_out[c] = 0.0;
  }

public:
  void set_kernel_args(const KernelArgs& args) override {
    q_reset_ = args.q_reset;
    feat_one_1_idx_ = args.feat_one_1_idx;
    feat_one_2_idx_ = args.feat_one_2_idx;
    feat_one_3_idx_ = args.feat_one_3_idx;
    feat_two_1_idx_ = args.feat_two_1_idx;
    feat_two_2_idx_ = args.feat_two_2_idx;
    feat_two_3_idx_ = args.feat_two_3_idx;
    arg_n_elem_ = args.n_elem;
  }

  void reset() override {
    BaseKernel::reset();
    q_mat_.clear();
    n_elem_ = 0;
    n_covs_ = 0;
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 5) {
               Rcpp::stop("RwSatlinkGammaCardKernel expects 5 parameter columns (q0, alpha, sat_single, sat_double, gamma), got %d",
                          (int)kernel_pars.cols.size());
             }

             const int n_comp = static_cast<int>(comp_idx.size());
             n_covs_ = covariate.ncol;
             n_elem_ = arg_n_elem_;

             if (n_comp == 0 || n_covs_ == 0) {
               q_mat_.clear();
               mark_run_complete();
               return;
             }

             if (n_elem_ < 2) {
               Rcpp::stop("RwSatlinkGammaCardKernel requires kernel_args$n_elem >= 2 to form a directional difference; got n_elem=%d",
                          n_elem_);
             }
             if (n_covs_ != n_elem_) {
               Rcpp::stop("RwSatlinkGammaCardKernel requires the number of covariate columns (%d) to equal kernel_args$n_elem (%d)",
                          n_covs_, n_elem_);
             }
             if (!feat_one_1_idx_ || !feat_two_1_idx_) {
               Rcpp::stop("RwSatlinkGammaCardKernel requires kernel_args$feat_one_1_idx_column and kernel_args$feat_two_1_idx_column");
             }

             const double* q0_col         = kernel_pars.cols[0];
             const double* alpha_col      = kernel_pars.cols[1];
             const double* sat_single_col = kernel_pars.cols[2];
             const double* sat_double_col = kernel_pars.cols[3];
             const double* gamma_col      = kernel_pars.cols[4];

             const int row0 = comp_idx[0];
             std::vector<double> q(n_elem_, q0_col[row0]);
             q_mat_.assign(n_comp * n_elem_, 0.0);

             emit_row(0, row0, sat_single_col, sat_double_col, gamma_col, q);

             for (int j = 0; j < n_comp - 1; ++j) {
               const int r = comp_idx[j];

               if (q_reset_ && q_reset_[r] == 1) {
                 std::fill(q.begin(), q.end(), q0_col[r]);
                 emit_row(j, r, sat_single_col, sat_double_col, gamma_col, q);
               }

               // Rescorla-Wagner summed-error update over the active (non-NA cov) cues:
               // one shared PE = reward - sum(active Q) updates every active cue.
               double reward     = NA_REAL;
               double q_active   = 0.0;
               bool   any_active = false;
               for (int e = 0; e < n_elem_; ++e) {
                 const double x = covariate(r, e);
                 if (!is_nan(x)) {
                   reward     = x;   // shared across all active cues
                   q_active  += q[e];
                   any_active = true;
                 }
               }
               if (any_active && !is_nan(reward)) {
                 const double alpha       = alpha_col[r];
                 const double compound_pe = reward - q_active;
                 for (int e = 0; e < n_elem_; ++e) {
                   if (!is_nan(covariate(r, e))) {
                     q[e] += alpha * compound_pe;
                   }
                 }
               }

               const int next_r = comp_idx[j + 1];
               emit_row(j + 1, next_r, sat_single_col, sat_double_col, gamma_col, q);
             }

             mark_run_complete();
           }

  KernelOutput get_output_stream(int code) const override {
    if (code != 1) {
      Rcpp::stop("RwSatlinkGammaCardKernel::get_output_stream: unsupported code %d (1=AllocShape)", code);
    }

    if (n_elem_ <= 0) return KernelOutput{ q_mat_.data(), 0, 0 };

    const int n_comp = static_cast<int>(q_mat_.size()) / n_elem_;
    std::vector<double>& buf = stream_buf_[0];

    if (has_expand_idx_) {
      const int n_full = static_cast<int>(expand_idx_.size());
      buf.resize(n_full * n_elem_);

      for (int c = 0; c < n_elem_; ++c) {
        for (int i = 0; i < n_full; ++i) {
          int comp_row = expand_idx_[i] - 1;
          buf[c * n_full + i] = q_mat_[comp_row * n_elem_ + c];
        }
      }

      return KernelOutput{ buf.data(), n_full, n_elem_ };
    }

    buf.resize(n_comp * n_elem_);

    for (int c = 0; c < n_elem_; ++c) {
      for (int r = 0; r < n_comp; ++r) {
        buf[c * n_comp + r] = q_mat_[r * n_elem_ + c];
      }
    }

    return KernelOutput{ buf.data(), n_comp, n_elem_ };
  }

  std::string output_stream_name(int code) const override {
    if (code == 1) return "AllocShape";
    throw std::runtime_error("RwSatlinkGammaCardKernel::output_stream_name: unsupported code");
  }
};


// =============================================================================
// RwSatlinkGammaCardPhKernel — CARDINALITY-saturation satlink read-out with
// RESCORLA-WAGNER (shared/summed PE) learning AND a PEARCE-HALL dynamic learning
// rate (per-cue "associability"). = RwSatlinkGammaCardKernel but the FIXED alpha
// is replaced by a per-cue associability state that relaxes toward the absolute
// shared prediction error (surprise-driven learning):
//   compound_pe = reward - sum(active Q over the chosen option's cues)
//   for each active cue e:
//     q[e]     += assoc[e] * compound_pe          // value update at rate assoc[e]
//     assoc[e]  = eta*|compound_pe| + (1-eta)*assoc[e]   // Pearce-Hall associability
// Cues that are inactive (NA covariate) carry both Q and associability forward.
// The combined {elem, cfg} cue set is supported via the optional 3rd cue per option
// (feat_*_3), exactly like RwSatlinkGammaCardKernel; cardinality (sat_single vs
// sat_double) is set by the ELEMENTAL second feature only. associability is the
// learning rate, so it is clamped to [0,1] after each update (the shared-PE |PE|
// can exceed 1 over >=2 cues, which without the clamp would let the rate run away;
// the cardsat tanh read-out also bounds the directional output). q0 = plogis,
// alpha0/eta = pnorm in (0,1), sat_*/gamma = exp. alpha0/eta -> stable values nest
// a constant-rate RW. Parameters: q0, alpha0, eta, sat_single, sat_double, gamma.
// =============================================================================
struct RwSatlinkGammaCardPhKernel : SequentialKernel {
private:
  int n_elem_ = 0;
  int n_covs_ = 0;

  const int* q_reset_ = nullptr;
  const int* feat_one_1_idx_ = nullptr;
  const int* feat_one_2_idx_ = nullptr;
  const int* feat_one_3_idx_ = nullptr;
  const int* feat_two_1_idx_ = nullptr;
  const int* feat_two_2_idx_ = nullptr;
  const int* feat_two_3_idx_ = nullptr;
  int arg_n_elem_ = -1;

  // Compressed [n_comp x n_elem_] allocation shape, row-major: [g, -g, 0, ...]
  std::vector<double> q_mat_;

  static double sat_link(double d, double theta, double gamma) {
    double dpow = d;
    if (!is_nan(gamma) && gamma > 0.0 && gamma != 1.0) {
      const double ad = std::fabs(d);
      dpow = (d < 0.0 ? -1.0 : 1.0) * std::pow(ad, gamma);
    }
    if (is_nan(theta) || theta <= 0.0) return dpow;
    if (theta > 1e6) return dpow;     // saturation off; avoid 0*inf overflow
    return theta * std::tanh(dpow / theta);
  }

  int required_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) {
      Rcpp::stop("RwSatlinkGammaCardPhKernel requires kernel_args$%s", arg_name);
    }
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) {
      Rcpp::stop("RwSatlinkGammaCardPhKernel requires kernel_args$%s[%d] to be a positive 1-based index",
                 arg_name, row + 1);
    }
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("RwSatlinkGammaCardPhKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  int optional_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) return -1;
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) return -1;
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("RwSatlinkGammaCardPhKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  void emit_row(int j, int r,
                const double* sat_single_col, const double* sat_double_col,
                const double* gamma_col,
                const std::vector<double>& q) {
    const int i11 = required_feat_index(feat_one_1_idx_, r, "feat_one_1_idx_column");
    const int i12 = optional_feat_index(feat_one_2_idx_, r, "feat_one_2_idx_column");
    const int i13 = optional_feat_index(feat_one_3_idx_, r, "feat_one_3_idx_column");
    const int i21 = required_feat_index(feat_two_1_idx_, r, "feat_two_1_idx_column");
    const int i22 = optional_feat_index(feat_two_2_idx_, r, "feat_two_2_idx_column");
    const int i23 = optional_feat_index(feat_two_3_idx_, r, "feat_two_3_idx_column");

    const double q11 = q[i11];
    const double q12 = (i12 >= 0) ? q[i12] : 0.0;
    const double q13 = (i13 >= 0) ? q[i13] : 0.0;
    const double q21 = q[i21];
    const double q22 = (i22 >= 0) ? q[i22] : 0.0;
    const double q23 = (i23 >= 0) ? q[i23] : 0.0;

    const double d = (q11 + q12 + q13) - (q21 + q22 + q23);

    const bool two_item = (i12 >= 0) || (i22 >= 0);
    const double sat = two_item ? sat_double_col[r] : sat_single_col[r];
    const double g = sat_link(d, sat, gamma_col[r]);

    double* row_out = &q_mat_[j * n_elem_];
    row_out[0] =  g;   // contribution when lR == UV_One
    row_out[1] = -g;   // contribution when lR == UV_Two
    for (int c = 2; c < n_elem_; ++c) row_out[c] = 0.0;
  }

public:
  void set_kernel_args(const KernelArgs& args) override {
    q_reset_ = args.q_reset;
    feat_one_1_idx_ = args.feat_one_1_idx;
    feat_one_2_idx_ = args.feat_one_2_idx;
    feat_one_3_idx_ = args.feat_one_3_idx;
    feat_two_1_idx_ = args.feat_two_1_idx;
    feat_two_2_idx_ = args.feat_two_2_idx;
    feat_two_3_idx_ = args.feat_two_3_idx;
    arg_n_elem_ = args.n_elem;
  }

  void reset() override {
    BaseKernel::reset();
    q_mat_.clear();
    n_elem_ = 0;
    n_covs_ = 0;
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 6) {
               Rcpp::stop("RwSatlinkGammaCardPhKernel expects 6 parameter columns (q0, alpha0, eta, sat_single, sat_double, gamma), got %d",
                          (int)kernel_pars.cols.size());
             }

             const int n_comp = static_cast<int>(comp_idx.size());
             n_covs_ = covariate.ncol;
             n_elem_ = arg_n_elem_;

             if (n_comp == 0 || n_covs_ == 0) {
               q_mat_.clear();
               mark_run_complete();
               return;
             }

             if (n_elem_ < 2) {
               Rcpp::stop("RwSatlinkGammaCardPhKernel requires kernel_args$n_elem >= 2 to form a directional difference; got n_elem=%d",
                          n_elem_);
             }
             if (n_covs_ != n_elem_) {
               Rcpp::stop("RwSatlinkGammaCardPhKernel requires the number of covariate columns (%d) to equal kernel_args$n_elem (%d)",
                          n_covs_, n_elem_);
             }
             if (!feat_one_1_idx_ || !feat_two_1_idx_) {
               Rcpp::stop("RwSatlinkGammaCardPhKernel requires kernel_args$feat_one_1_idx_column and kernel_args$feat_two_1_idx_column");
             }

             const double* q0_col         = kernel_pars.cols[0];
             const double* alpha0_col     = kernel_pars.cols[1];
             const double* eta_col        = kernel_pars.cols[2];
             const double* sat_single_col = kernel_pars.cols[3];
             const double* sat_double_col = kernel_pars.cols[4];
             const double* gamma_col      = kernel_pars.cols[5];

             const int row0 = comp_idx[0];
             std::vector<double> q(n_elem_, q0_col[row0]);
             std::vector<double> assoc(n_elem_, alpha0_col[row0]);  // per-cue associability (PH rate)
             q_mat_.assign(n_comp * n_elem_, 0.0);

             emit_row(0, row0, sat_single_col, sat_double_col, gamma_col, q);

             for (int j = 0; j < n_comp - 1; ++j) {
               const int r = comp_idx[j];

               if (q_reset_ && q_reset_[r] == 1) {
                 std::fill(q.begin(), q.end(), q0_col[r]);
                 std::fill(assoc.begin(), assoc.end(), alpha0_col[r]);  // reset associability too
                 emit_row(j, r, sat_single_col, sat_double_col, gamma_col, q);
               }

               // Rescorla-Wagner summed-error prediction over the active (non-NA cov) cues:
               // one shared compound PE = reward - sum(active Q). Each active cue updates at
               // ITS OWN current associability and then relaxes that associability toward
               // |compound_pe| (Pearce-Hall surprise-driven rate).
               double reward     = NA_REAL;
               double q_active   = 0.0;
               bool   any_active = false;
               for (int e = 0; e < n_elem_; ++e) {
                 const double x = covariate(r, e);
                 if (!is_nan(x)) {
                   reward     = x;   // shared across all active cues
                   q_active  += q[e];
                   any_active = true;
                 }
               }
               if (any_active && !is_nan(reward)) {
                 const double compound_pe = reward - q_active;
                 const double abs_pe      = std::fabs(compound_pe);
                 double eta               = eta_col[r];
                 if (is_nan(eta)) eta = 0.0;
                 if (eta < 0.0) eta = 0.0;
                 if (eta > 1.0) eta = 1.0;
                 for (int e = 0; e < n_elem_; ++e) {
                   if (!is_nan(covariate(r, e))) {
                     q[e] += assoc[e] * compound_pe;         // rate = current associability
                     double a = eta * abs_pe + (1.0 - eta) * assoc[e];
                     if (a < 0.0) a = 0.0;
                     if (a > 1.0) a = 1.0;                   // clamp: assoc IS the LR -> keep in [0,1]
                     assoc[e] = a;
                   }
                 }
               }

               const int next_r = comp_idx[j + 1];
               emit_row(j + 1, next_r, sat_single_col, sat_double_col, gamma_col, q);
             }

             mark_run_complete();
           }

  KernelOutput get_output_stream(int code) const override {
    if (code != 1) {
      Rcpp::stop("RwSatlinkGammaCardPhKernel::get_output_stream: unsupported code %d (1=AllocShape)", code);
    }

    if (n_elem_ <= 0) return KernelOutput{ q_mat_.data(), 0, 0 };

    const int n_comp = static_cast<int>(q_mat_.size()) / n_elem_;
    std::vector<double>& buf = stream_buf_[0];

    if (has_expand_idx_) {
      const int n_full = static_cast<int>(expand_idx_.size());
      buf.resize(n_full * n_elem_);

      for (int c = 0; c < n_elem_; ++c) {
        for (int i = 0; i < n_full; ++i) {
          int comp_row = expand_idx_[i] - 1;
          buf[c * n_full + i] = q_mat_[comp_row * n_elem_ + c];
        }
      }

      return KernelOutput{ buf.data(), n_full, n_elem_ };
    }

    buf.resize(n_comp * n_elem_);

    for (int c = 0; c < n_elem_; ++c) {
      for (int r = 0; r < n_comp; ++r) {
        buf[c * n_comp + r] = q_mat_[r * n_elem_ + c];
      }
    }

    return KernelOutput{ buf.data(), n_comp, n_elem_ };
  }

  std::string output_stream_name(int code) const override {
    if (code == 1) return "AllocShape";
    throw std::runtime_error("RwSatlinkGammaCardPhKernel::output_stream_name: unsupported code");
  }
};


// =============================================================================
// RwSatlinkGammaCardExpdecrKernel — CARDINALITY-saturation satlink read-out with
// RESCORLA-WAGNER (shared/summed PE) learning AND an exp-DECREASING learning rate.
// = RwSatlinkGammaCardKernel + the DeltaExpdecr schedule: the FIXED alpha is
// replaced by a deterministic per-trial rate
//   alpha_eff(t) = Phi(alpha_base + alpha_w * exp(-d_alpha_ed * block_trial[r]))
// (alpha_base on probit scale; alpha_w, d_alpha_ed exp-transformed >= 0; block_trial
// = within-block 0-based exposure count). The same shared compound PE updates all
// active cues at alpha_eff(t). d_alpha_ed -> 0 or alpha_w -> 0 nests the constant-rate
// RwSatlinkGammaCardKernel. Combined {elem, cfg} cue set via the optional 3rd cue per
// option (feat_*_3), like RwSatlinkGammaCardKernel; cardinality from the elemental
// second feature only. Parameters: q0, alpha_base, alpha_w, d_alpha_ed, sat_single,
// sat_double, gamma.
// =============================================================================
struct RwSatlinkGammaCardExpdecrKernel : SequentialKernel {
private:
  int n_elem_ = 0;
  int n_covs_ = 0;

  const int* q_reset_ = nullptr;
  const int* feat_one_1_idx_ = nullptr;
  const int* feat_one_2_idx_ = nullptr;
  const int* feat_one_3_idx_ = nullptr;
  const int* feat_two_1_idx_ = nullptr;
  const int* feat_two_2_idx_ = nullptr;
  const int* feat_two_3_idx_ = nullptr;
  const double* block_trial_ = nullptr;
  int arg_n_elem_ = -1;

  // Compressed [n_comp x n_elem_] allocation shape, row-major: [g, -g, 0, ...]
  std::vector<double> q_mat_;

  static double sat_link(double d, double theta, double gamma) {
    double dpow = d;
    if (!is_nan(gamma) && gamma > 0.0 && gamma != 1.0) {
      const double ad = std::fabs(d);
      dpow = (d < 0.0 ? -1.0 : 1.0) * std::pow(ad, gamma);
    }
    if (is_nan(theta) || theta <= 0.0) return dpow;
    if (theta > 1e6) return dpow;     // saturation off; avoid 0*inf overflow
    return theta * std::tanh(dpow / theta);
  }

  double normal_cdf(double x) const {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
  }

  // alpha_eff(t) = Phi(alpha_base + alpha_w * exp(-d_alpha_ed * block_trial[r])).
  double learning_rate(int r,
                       const double* alpha_base_col,
                       const double* alpha_w_col,
                       const double* d_alpha_ed_col) const {
    double t = block_trial_ ? block_trial_[r] : 0.0;
    if (is_nan(t) || t < 0.0) t = 0.0;
    double d_alpha_ed = d_alpha_ed_col[r];
    if (is_nan(d_alpha_ed) || d_alpha_ed < 0.0) d_alpha_ed = 0.0;
    double alpha_base = alpha_base_col[r];
    if (is_nan(alpha_base)) alpha_base = 0.0;
    double alpha_w = alpha_w_col[r];
    if (is_nan(alpha_w) || alpha_w < 0.0) alpha_w = 0.0;
    return normal_cdf(alpha_base + alpha_w * std::exp(-d_alpha_ed * t));
  }

  int required_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) {
      Rcpp::stop("RwSatlinkGammaCardExpdecrKernel requires kernel_args$%s", arg_name);
    }
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) {
      Rcpp::stop("RwSatlinkGammaCardExpdecrKernel requires kernel_args$%s[%d] to be a positive 1-based index",
                 arg_name, row + 1);
    }
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("RwSatlinkGammaCardExpdecrKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  int optional_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) return -1;
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) return -1;
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("RwSatlinkGammaCardExpdecrKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  void emit_row(int j, int r,
                const double* sat_single_col, const double* sat_double_col,
                const double* gamma_col,
                const std::vector<double>& q) {
    const int i11 = required_feat_index(feat_one_1_idx_, r, "feat_one_1_idx_column");
    const int i12 = optional_feat_index(feat_one_2_idx_, r, "feat_one_2_idx_column");
    const int i13 = optional_feat_index(feat_one_3_idx_, r, "feat_one_3_idx_column");
    const int i21 = required_feat_index(feat_two_1_idx_, r, "feat_two_1_idx_column");
    const int i22 = optional_feat_index(feat_two_2_idx_, r, "feat_two_2_idx_column");
    const int i23 = optional_feat_index(feat_two_3_idx_, r, "feat_two_3_idx_column");

    const double q11 = q[i11];
    const double q12 = (i12 >= 0) ? q[i12] : 0.0;
    const double q13 = (i13 >= 0) ? q[i13] : 0.0;
    const double q21 = q[i21];
    const double q22 = (i22 >= 0) ? q[i22] : 0.0;
    const double q23 = (i23 >= 0) ? q[i23] : 0.0;

    const double d = (q11 + q12 + q13) - (q21 + q22 + q23);

    const bool two_item = (i12 >= 0) || (i22 >= 0);
    const double sat = two_item ? sat_double_col[r] : sat_single_col[r];
    const double g = sat_link(d, sat, gamma_col[r]);

    double* row_out = &q_mat_[j * n_elem_];
    row_out[0] =  g;   // contribution when lR == UV_One
    row_out[1] = -g;   // contribution when lR == UV_Two
    for (int c = 2; c < n_elem_; ++c) row_out[c] = 0.0;
  }

public:
  void set_kernel_args(const KernelArgs& args) override {
    q_reset_ = args.q_reset;
    feat_one_1_idx_ = args.feat_one_1_idx;
    feat_one_2_idx_ = args.feat_one_2_idx;
    feat_one_3_idx_ = args.feat_one_3_idx;
    feat_two_1_idx_ = args.feat_two_1_idx;
    feat_two_2_idx_ = args.feat_two_2_idx;
    feat_two_3_idx_ = args.feat_two_3_idx;
    block_trial_ = args.block_trial;
    arg_n_elem_ = args.n_elem;
  }

  void reset() override {
    BaseKernel::reset();
    q_mat_.clear();
    n_elem_ = 0;
    n_covs_ = 0;
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 7) {
               Rcpp::stop("RwSatlinkGammaCardExpdecrKernel expects 7 parameter columns (q0, alpha_base, alpha_w, d_alpha_ed, sat_single, sat_double, gamma), got %d",
                          (int)kernel_pars.cols.size());
             }

             const int n_comp = static_cast<int>(comp_idx.size());
             n_covs_ = covariate.ncol;
             n_elem_ = arg_n_elem_;

             if (n_comp == 0 || n_covs_ == 0) {
               q_mat_.clear();
               mark_run_complete();
               return;
             }

             if (n_elem_ < 2) {
               Rcpp::stop("RwSatlinkGammaCardExpdecrKernel requires kernel_args$n_elem >= 2 to form a directional difference; got n_elem=%d",
                          n_elem_);
             }
             if (n_covs_ != n_elem_) {
               Rcpp::stop("RwSatlinkGammaCardExpdecrKernel requires the number of covariate columns (%d) to equal kernel_args$n_elem (%d)",
                          n_covs_, n_elem_);
             }
             if (!feat_one_1_idx_ || !feat_two_1_idx_) {
               Rcpp::stop("RwSatlinkGammaCardExpdecrKernel requires kernel_args$feat_one_1_idx_column and kernel_args$feat_two_1_idx_column");
             }
             if (!block_trial_) {
               Rcpp::stop("RwSatlinkGammaCardExpdecrKernel requires kernel_args$block_trial_column");
             }

             const double* q0_col         = kernel_pars.cols[0];
             const double* alpha_base_col = kernel_pars.cols[1];
             const double* alpha_w_col    = kernel_pars.cols[2];
             const double* d_alpha_ed_col = kernel_pars.cols[3];
             const double* sat_single_col = kernel_pars.cols[4];
             const double* sat_double_col = kernel_pars.cols[5];
             const double* gamma_col      = kernel_pars.cols[6];

             const int row0 = comp_idx[0];
             std::vector<double> q(n_elem_, q0_col[row0]);
             q_mat_.assign(n_comp * n_elem_, 0.0);

             emit_row(0, row0, sat_single_col, sat_double_col, gamma_col, q);

             for (int j = 0; j < n_comp - 1; ++j) {
               const int r = comp_idx[j];

               if (q_reset_ && q_reset_[r] == 1) {
                 std::fill(q.begin(), q.end(), q0_col[r]);
                 emit_row(j, r, sat_single_col, sat_double_col, gamma_col, q);
               }

               // RW summed-error update at the exp-decreasing rate alpha_eff(t).
               const double alpha = learning_rate(r, alpha_base_col, alpha_w_col, d_alpha_ed_col);
               double reward     = NA_REAL;
               double q_active   = 0.0;
               bool   any_active = false;
               for (int e = 0; e < n_elem_; ++e) {
                 const double x = covariate(r, e);
                 if (!is_nan(x)) {
                   reward     = x;   // shared across all active cues
                   q_active  += q[e];
                   any_active = true;
                 }
               }
               if (any_active && !is_nan(reward)) {
                 const double compound_pe = reward - q_active;
                 for (int e = 0; e < n_elem_; ++e) {
                   if (!is_nan(covariate(r, e))) {
                     q[e] += alpha * compound_pe;
                   }
                 }
               }

               const int next_r = comp_idx[j + 1];
               emit_row(j + 1, next_r, sat_single_col, sat_double_col, gamma_col, q);
             }

             mark_run_complete();
           }

  KernelOutput get_output_stream(int code) const override {
    if (code != 1) {
      Rcpp::stop("RwSatlinkGammaCardExpdecrKernel::get_output_stream: unsupported code %d (1=AllocShape)", code);
    }

    if (n_elem_ <= 0) return KernelOutput{ q_mat_.data(), 0, 0 };

    const int n_comp = static_cast<int>(q_mat_.size()) / n_elem_;
    std::vector<double>& buf = stream_buf_[0];

    if (has_expand_idx_) {
      const int n_full = static_cast<int>(expand_idx_.size());
      buf.resize(n_full * n_elem_);

      for (int c = 0; c < n_elem_; ++c) {
        for (int i = 0; i < n_full; ++i) {
          int comp_row = expand_idx_[i] - 1;
          buf[c * n_full + i] = q_mat_[comp_row * n_elem_ + c];
        }
      }

      return KernelOutput{ buf.data(), n_full, n_elem_ };
    }

    buf.resize(n_comp * n_elem_);

    for (int c = 0; c < n_elem_; ++c) {
      for (int r = 0; r < n_comp; ++r) {
        buf[c * n_comp + r] = q_mat_[r * n_elem_ + c];
      }
    }

    return KernelOutput{ buf.data(), n_comp, n_elem_ };
  }

  std::string output_stream_name(int code) const override {
    if (code == 1) return "AllocShape";
    throw std::runtime_error("RwSatlinkGammaCardExpdecrKernel::output_stream_name: unsupported code");
  }
};


// =============================================================================
// RwSatlinkGammaCardFfKernel — FULL-FEEDBACK Rescorla-Wagner cardsat read-out.
// Same cardsat read-out as RwSatlinkGammaCardKernel, but the LEARNING does TWO
// per-option summed-error updates per trial (the design shows BOTH options'
// outcomes every trial):
//   PE_one = r_one - (q[one_1] + q[one_2] + q[one_3]); update those by alpha*PE_one
//   PE_two = r_two - (q[two_1] + q[two_2] + q[two_3]); update those by alpha*PE_two
// where r_one / r_two are read from the covariate columns of each option's first
// cue (the full-feedback covariate encoding sets every present cue's column to its
// OWN option's reward). Cues are grouped by the feat_one_* / feat_two_* index
// columns (the configural compound rides as the optional 3rd cue). Single Q per cue
// (screen-side is fixed in this design, so no Q-splitting). q0, alpha, sat_single,
// sat_double, gamma -- RW adds no params; sat -> Inf nests the linear read-out.
// =============================================================================
struct RwSatlinkGammaCardFfKernel : SequentialKernel {
private:
  int n_elem_ = 0;
  int n_covs_ = 0;

  const int* q_reset_ = nullptr;
  const int* feat_one_1_idx_ = nullptr;
  const int* feat_one_2_idx_ = nullptr;
  const int* feat_one_3_idx_ = nullptr;
  const int* feat_two_1_idx_ = nullptr;
  const int* feat_two_2_idx_ = nullptr;
  const int* feat_two_3_idx_ = nullptr;
  int arg_n_elem_ = -1;

  std::vector<double> q_mat_;

  static double sat_link(double d, double theta, double gamma) {
    double dpow = d;
    if (!is_nan(gamma) && gamma > 0.0 && gamma != 1.0) {
      const double ad = std::fabs(d);
      dpow = (d < 0.0 ? -1.0 : 1.0) * std::pow(ad, gamma);
    }
    if (is_nan(theta) || theta <= 0.0) return dpow;
    if (theta > 1e6) return dpow;
    return theta * std::tanh(dpow / theta);
  }

  int required_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) {
      Rcpp::stop("RwSatlinkGammaCardFfKernel requires kernel_args$%s", arg_name);
    }
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) {
      Rcpp::stop("RwSatlinkGammaCardFfKernel requires kernel_args$%s[%d] to be a positive 1-based index",
                 arg_name, row + 1);
    }
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("RwSatlinkGammaCardFfKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  int optional_feat_index(const int* arr, int row, const char* arg_name) const {
    if (!arr) return -1;
    const int value = arr[row];
    if (value == NA_INTEGER || value <= 0) return -1;
    const int idx = value - 1;
    if (idx < 0 || idx >= n_elem_) {
      Rcpp::stop("RwSatlinkGammaCardFfKernel kernel_args$%s[%d]=%d is outside 1:n_elem (%d)",
                 arg_name, row + 1, value, n_elem_);
    }
    return idx;
  }

  void emit_row(int j, int r,
                const double* sat_single_col, const double* sat_double_col,
                const double* gamma_col,
                const std::vector<double>& q) {
    const int i11 = required_feat_index(feat_one_1_idx_, r, "feat_one_1_idx_column");
    const int i12 = optional_feat_index(feat_one_2_idx_, r, "feat_one_2_idx_column");
    const int i13 = optional_feat_index(feat_one_3_idx_, r, "feat_one_3_idx_column");
    const int i21 = required_feat_index(feat_two_1_idx_, r, "feat_two_1_idx_column");
    const int i22 = optional_feat_index(feat_two_2_idx_, r, "feat_two_2_idx_column");
    const int i23 = optional_feat_index(feat_two_3_idx_, r, "feat_two_3_idx_column");

    const double q11 = q[i11];
    const double q12 = (i12 >= 0) ? q[i12] : 0.0;
    const double q13 = (i13 >= 0) ? q[i13] : 0.0;
    const double q21 = q[i21];
    const double q22 = (i22 >= 0) ? q[i22] : 0.0;
    const double q23 = (i23 >= 0) ? q[i23] : 0.0;

    const double d = (q11 + q12 + q13) - (q21 + q22 + q23);

    const bool two_item = (i12 >= 0) || (i22 >= 0);
    const double sat = two_item ? sat_double_col[r] : sat_single_col[r];
    const double g = sat_link(d, sat, gamma_col[r]);

    double* row_out = &q_mat_[j * n_elem_];
    row_out[0] =  g;
    row_out[1] = -g;
    for (int c = 2; c < n_elem_; ++c) row_out[c] = 0.0;
  }

public:
  void set_kernel_args(const KernelArgs& args) override {
    q_reset_ = args.q_reset;
    feat_one_1_idx_ = args.feat_one_1_idx;
    feat_one_2_idx_ = args.feat_one_2_idx;
    feat_one_3_idx_ = args.feat_one_3_idx;
    feat_two_1_idx_ = args.feat_two_1_idx;
    feat_two_2_idx_ = args.feat_two_2_idx;
    feat_two_3_idx_ = args.feat_two_3_idx;
    arg_n_elem_ = args.n_elem;
  }

  void reset() override {
    BaseKernel::reset();
    q_mat_.clear();
    n_elem_ = 0;
    n_covs_ = 0;
  }

  void run(const KernelParsView& kernel_pars,
           const Mat& covariate,
           const std::vector<int>& comp_idx) override {
             if (kernel_pars.cols.size() != 5) {
               Rcpp::stop("RwSatlinkGammaCardFfKernel expects 5 parameter columns (q0, alpha, sat_single, sat_double, gamma), got %d",
                          (int)kernel_pars.cols.size());
             }

             const int n_comp = static_cast<int>(comp_idx.size());
             n_covs_ = covariate.ncol;
             n_elem_ = arg_n_elem_;

             if (n_comp == 0 || n_covs_ == 0) {
               q_mat_.clear();
               mark_run_complete();
               return;
             }

             if (n_elem_ < 2) {
               Rcpp::stop("RwSatlinkGammaCardFfKernel requires kernel_args$n_elem >= 2 to form a directional difference; got n_elem=%d",
                          n_elem_);
             }
             if (n_covs_ != n_elem_) {
               Rcpp::stop("RwSatlinkGammaCardFfKernel requires the number of covariate columns (%d) to equal kernel_args$n_elem (%d)",
                          n_covs_, n_elem_);
             }
             if (!feat_one_1_idx_ || !feat_two_1_idx_) {
               Rcpp::stop("RwSatlinkGammaCardFfKernel requires kernel_args$feat_one_1_idx_column and kernel_args$feat_two_1_idx_column");
             }

             const double* q0_col         = kernel_pars.cols[0];
             const double* alpha_col      = kernel_pars.cols[1];
             const double* sat_single_col = kernel_pars.cols[2];
             const double* sat_double_col = kernel_pars.cols[3];
             const double* gamma_col      = kernel_pars.cols[4];

             const int row0 = comp_idx[0];
             std::vector<double> q(n_elem_, q0_col[row0]);
             q_mat_.assign(n_comp * n_elem_, 0.0);

             emit_row(0, row0, sat_single_col, sat_double_col, gamma_col, q);

             for (int j = 0; j < n_comp - 1; ++j) {
               const int r = comp_idx[j];

               if (q_reset_ && q_reset_[r] == 1) {
                 std::fill(q.begin(), q.end(), q0_col[r]);
                 emit_row(j, r, sat_single_col, sat_double_col, gamma_col, q);
               }

               // FULL-FEEDBACK RW: two per-option summed-error updates on the shared Q.
               const int u_i11 = required_feat_index(feat_one_1_idx_, r, "feat_one_1_idx_column");
               const int u_i12 = optional_feat_index(feat_one_2_idx_, r, "feat_one_2_idx_column");
               const int u_i13 = optional_feat_index(feat_one_3_idx_, r, "feat_one_3_idx_column");
               const int u_i21 = required_feat_index(feat_two_1_idx_, r, "feat_two_1_idx_column");
               const int u_i22 = optional_feat_index(feat_two_2_idx_, r, "feat_two_2_idx_column");
               const int u_i23 = optional_feat_index(feat_two_3_idx_, r, "feat_two_3_idx_column");
               const double alpha = alpha_col[r];

               // Option 1: reward from option-1's first cue's covariate column.
               const double r_one = covariate(r, u_i11);
               if (!is_nan(r_one)) {
                 double sum_one = q[u_i11] + (u_i12 >= 0 ? q[u_i12] : 0.0) + (u_i13 >= 0 ? q[u_i13] : 0.0);
                 double pe_one = r_one - sum_one;
                 q[u_i11] += alpha * pe_one;
                 if (u_i12 >= 0) q[u_i12] += alpha * pe_one;
                 if (u_i13 >= 0) q[u_i13] += alpha * pe_one;
               }
               // Option 2: reward from option-2's first cue's covariate column.
               const double r_two = covariate(r, u_i21);
               if (!is_nan(r_two)) {
                 double sum_two = q[u_i21] + (u_i22 >= 0 ? q[u_i22] : 0.0) + (u_i23 >= 0 ? q[u_i23] : 0.0);
                 double pe_two = r_two - sum_two;
                 q[u_i21] += alpha * pe_two;
                 if (u_i22 >= 0) q[u_i22] += alpha * pe_two;
                 if (u_i23 >= 0) q[u_i23] += alpha * pe_two;
               }

               const int next_r = comp_idx[j + 1];
               emit_row(j + 1, next_r, sat_single_col, sat_double_col, gamma_col, q);
             }

             mark_run_complete();
           }

  KernelOutput get_output_stream(int code) const override {
    if (code != 1) {
      Rcpp::stop("RwSatlinkGammaCardFfKernel::get_output_stream: unsupported code %d (1=AllocShape)", code);
    }

    if (n_elem_ <= 0) return KernelOutput{ q_mat_.data(), 0, 0 };

    const int n_comp = static_cast<int>(q_mat_.size()) / n_elem_;
    std::vector<double>& buf = stream_buf_[0];

    if (has_expand_idx_) {
      const int n_full = static_cast<int>(expand_idx_.size());
      buf.resize(n_full * n_elem_);
      for (int c = 0; c < n_elem_; ++c) {
        for (int i = 0; i < n_full; ++i) {
          int comp_row = expand_idx_[i] - 1;
          buf[c * n_full + i] = q_mat_[comp_row * n_elem_ + c];
        }
      }
      return KernelOutput{ buf.data(), n_full, n_elem_ };
    }

    buf.resize(n_comp * n_elem_);
    for (int c = 0; c < n_elem_; ++c) {
      for (int r = 0; r < n_comp; ++r) {
        buf[c * n_comp + r] = q_mat_[r * n_elem_ + c];
      }
    }
    return KernelOutput{ buf.data(), n_comp, n_elem_ };
  }

  std::string output_stream_name(int code) const override {
    if (code == 1) return "AllocShape";
    throw std::runtime_error("RwSatlinkGammaCardFfKernel::output_stream_name: unsupported code");
  }
};


// ---- Type mapping + factory ----

KernelType to_kernel_type(const Rcpp::String& k);

std::unique_ptr<BaseKernel> make_kernel(KernelType kt,
                                        SEXP custom_fun = R_NilValue);


#endif // KERNELS_H

