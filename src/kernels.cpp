#include "kernels.h"

KernelType to_kernel_type(const Rcpp::String& k) {
  if (k == "delta2kernel") return KernelType::Delta2Kernel;
  // if (k == "delta2kernel2") return KernelType::Delta2Kernel2;
  if (k == "delta")        return KernelType::SimpleDelta;
  if (k == "delta_plogis_q0") return KernelType::SimpleDelta;  // SimpleDelta + plogis q0 transform (R-side)
  if (k == "delta2lr")     return KernelType::Delta2LR;
  if (k == "lin_incr")     return KernelType::LinIncr;
  if (k == "lin_decr")     return KernelType::LinDecr;
  if (k == "exp_incr")     return KernelType::ExpIncr;
  if (k == "exp_decr")     return KernelType::ExpDecr;
  if (k == "pow_incr")     return KernelType::PowIncr;
  if (k == "pow_decr")     return KernelType::PowDecr;
  if (k == "poly2")        return KernelType::Poly2;
  if (k == "poly3")        return KernelType::Poly3;
  if (k == "poly4")        return KernelType::Poly4;
  if (k == "custom")       return KernelType::Custom;
  if (k == "rescorlawagner")       return KernelType::RescorlaWagner;
  if (k == "delta_satlink_gamma_card") return KernelType::DeltaSatlinkGammaCard;
  if (k == "delta_satlink_dim_card") return KernelType::DeltaSatlinkDimCard;
  if (k == "delta_noisyor_card") return KernelType::DeltaNoisyOrCard;
  if (k == "rw_noisyor_card") return KernelType::RwNoisyOrCard;
  if (k == "delta_satlink_gamma_card_expdecr") return KernelType::DeltaSatlinkGammaCardExpdecr;
  if (k == "delta_expdecr") return KernelType::DeltaExpdecr;
  if (k == "rw_satlink_gamma_card") return KernelType::RwSatlinkGammaCard;
  if (k == "rw_satlink_gamma_card_ff") return KernelType::RwSatlinkGammaCardFf;
  if (k == "rw_satlink_gamma_card_2ch") return KernelType::RwSatlinkGammaCard2ch;
  if (k == "rw_satlink_gamma_card_2ch_ff") return KernelType::RwSatlinkGammaCard2chFf;
  if (k == "rw_satlink_gamma_card_ph") return KernelType::RwSatlinkGammaCardPh;
  if (k == "rw_satlink_gamma_card_expdecr") return KernelType::RwSatlinkGammaCardExpdecr;
  if (k == "rw_satlink_dim_card") return KernelType::RwSatlinkDimCard;
  if (k == "rw_satlink_dim_card_ff") return KernelType::RwSatlinkDimCardFf;
  if (k == "beta_binomial")       return KernelType::BetaBinomial;
  if (k == "beta_binomial_decay") return KernelType::BetaBinomialDecay;
  if (k == "beta_binomial_window")return KernelType::BetaBinomialWindow;
  if (k == "dbm")                 return KernelType::DBM;
  if (k == "tpm")                 return KernelType::TPM;

  Rcpp::stop("Unknown kernel type");
}


std::unique_ptr<BaseKernel> make_kernel(KernelType kt, SEXP custom_fun) {
  switch (kt) {
  case KernelType::SimpleDelta: return std::unique_ptr<BaseKernel>(new SimpleDelta());
  case KernelType::Delta2Kernel: return std::unique_ptr<BaseKernel>(new Delta2Kernel());
  // case KernelType::Delta2Kernel2: return std::unique_ptr<BaseKernel>(new Delta2Kernel2());
  case KernelType::Delta2LR:    return std::unique_ptr<BaseKernel>(new Delta2LR());

  case KernelType::LinIncr:     return std::unique_ptr<BaseKernel>(new LinIncrKernel());
  case KernelType::LinDecr:     return std::unique_ptr<BaseKernel>(new LinDecrKernel());
  case KernelType::ExpIncr:     return std::unique_ptr<BaseKernel>(new ExpIncrKernel());
  case KernelType::ExpDecr:     return std::unique_ptr<BaseKernel>(new ExpDecrKernel());
  case KernelType::PowIncr:     return std::unique_ptr<BaseKernel>(new PowIncrKernel());
  case KernelType::PowDecr:     return std::unique_ptr<BaseKernel>(new PowDecrKernel());
  case KernelType::Poly2:       return std::unique_ptr<BaseKernel>(new Poly2Kernel());
  case KernelType::Poly3:       return std::unique_ptr<BaseKernel>(new Poly3Kernel());
  case KernelType::Poly4:       return std::unique_ptr<BaseKernel>(new Poly4Kernel());
  case KernelType::RescorlaWagner:     return std::unique_ptr<BaseKernel>(new RescorlaWagnerKernel());
  case KernelType::DeltaSatlinkGammaCard: return std::unique_ptr<BaseKernel>(new DeltaSatlinkGammaCardKernel());
  case KernelType::DeltaSatlinkDimCard: return std::unique_ptr<BaseKernel>(new DeltaSatlinkDimCardKernel());
  case KernelType::DeltaNoisyOrCard: return std::unique_ptr<BaseKernel>(new DeltaNoisyOrCardKernel());
  case KernelType::RwNoisyOrCard: return std::unique_ptr<BaseKernel>(new RwNoisyOrCardKernel());
  case KernelType::DeltaSatlinkGammaCardExpdecr: return std::unique_ptr<BaseKernel>(new DeltaSatlinkGammaCardExpdecrKernel());
  case KernelType::DeltaExpdecr: return std::unique_ptr<BaseKernel>(new DeltaExpdecrKernel());
  case KernelType::RwSatlinkGammaCard: return std::unique_ptr<BaseKernel>(new RwSatlinkGammaCardKernel());
  case KernelType::RwSatlinkGammaCardFf: return std::unique_ptr<BaseKernel>(new RwSatlinkGammaCardFfKernel());
  case KernelType::RwSatlinkGammaCard2ch: return std::unique_ptr<BaseKernel>(new RwSatlinkGammaCard2chKernel());
  case KernelType::RwSatlinkGammaCard2chFf: return std::unique_ptr<BaseKernel>(new RwSatlinkGammaCard2chFfKernel());
  case KernelType::RwSatlinkGammaCardPh: return std::unique_ptr<BaseKernel>(new RwSatlinkGammaCardPhKernel());
  case KernelType::RwSatlinkGammaCardExpdecr: return std::unique_ptr<BaseKernel>(new RwSatlinkGammaCardExpdecrKernel());
  case KernelType::RwSatlinkDimCard: return std::unique_ptr<BaseKernel>(new RwSatlinkDimCardKernel());
  case KernelType::RwSatlinkDimCardFf: return std::unique_ptr<BaseKernel>(new RwSatlinkDimCardFfKernel());
  case KernelType::BetaBinomial:       return std::make_unique<BetaBinomialKernel>();
  case KernelType::BetaBinomialDecay:  return std::make_unique<BetaBinomialDecayKernel>();
  case KernelType::BetaBinomialWindow: return std::make_unique<BetaBinomialWindowKernel>();
  case KernelType::DBM:                return std::make_unique<DBMKernel>();
  case KernelType::TPM:                return std::make_unique<TPMKernel>();

  case KernelType::Custom:
    if (custom_fun == R_NilValue) {
      Rcpp::stop("make_kernel: Custom kernel requested but custom_fun is NULL");
    }
    return std::unique_ptr<BaseKernel>(new CustomKernel(custom_fun));
  }

  Rcpp::stop("Unknown kernel type");
}
