#ifndef FASTCPD_CLASS_H_
#define FASTCPD_CLASS_H_

#include <armadillo>
#include <optional>

namespace fastcpd {

namespace classes {

class Fastcpd {
 public:
  Fastcpd(double const beta, std::string const& cost_adjustment,
          bool const cp_only, arma::mat const& data, double const epsilon,
          std::string const& family, arma::colvec const& line_search,
          arma::colvec const& lower, double const momentum_coef,
          arma::colvec const& order, int const p, unsigned int const p_response,
          double const pruning_coef, int const segment_count, double const trim,
          arma::colvec const& upper, double const vanilla_percentage,
          arma::mat const& variance_estimate, bool const warm_start);

  std::tuple<arma::colvec, arma::colvec, arma::colvec, arma::mat, arma::mat>
  Run();

 private:
  struct FunctionSet {
    arma::colvec (Fastcpd::*gradient)(unsigned int const segment_start,
                                      unsigned int const segment_end,
                                      arma::colvec const& theta);
    arma::mat (Fastcpd::*hessian)(unsigned int const segment_start,
                                  unsigned int const segment_end,
                                  arma::colvec const& theta);
    void (Fastcpd::*nll_pelt)(unsigned int const segment_start,
                              unsigned int const segment_end, bool const cv,
                              std::optional<arma::colvec> const& start);
    void (Fastcpd::*nll_pelt_value)(unsigned int const segment_start,
                                    unsigned int const segment_end,
                                    bool const cv,
                                    std::optional<arma::colvec> const& start);
    double (Fastcpd::*nll_sen)(unsigned int const segment_start,
                               unsigned int const segment_end,
                               arma::colvec const& theta);
  };

  void CreateSenParameters();
  void CreateSegmentStatistics();
  double GetCostAdjustmentValue(unsigned int const nrows);
  void GetCostResult(unsigned int const segment_start,
                     unsigned int const segment_end,
                     std::optional<arma::colvec> theta, bool const cv = false,
                     std::optional<arma::colvec> start = std::nullopt);
  std::tuple<arma::colvec, arma::colvec, arma::colvec, arma::mat, arma::mat>
  GetChangePointSet();
  double GetCostValue(int const tau, unsigned int const i);
  void GetCostValuePelt(unsigned int const segment_start,
                        unsigned int const segment_end, unsigned int const i);
  double GetCostValueSen(unsigned int const segment_start,
                         unsigned int const segment_end, unsigned int const i);
  arma::colvec GetGradientArma(unsigned int const segment_start,
                               unsigned int const segment_end,
                               arma::colvec const& theta);
  arma::colvec GetGradientBinomial(unsigned int const segment_start,
                                   unsigned int const segment_end,
                                   arma::colvec const& theta);
  arma::colvec GetGradientCustom(unsigned int const segment_start,
                                 unsigned int const segment_end,
                                 arma::colvec const& theta);
  arma::colvec GetGradientLm(unsigned int const segment_start,
                             unsigned int const segment_end,
                             arma::colvec const& theta);
  arma::colvec GetGradientMa(unsigned int const segment_start,
                             unsigned int const segment_end,
                             arma::colvec const& theta);
  arma::colvec GetGradientPoisson(unsigned int const segment_start,
                                  unsigned int const segment_end,
                                  arma::colvec const& theta);
  arma::mat GetHessianArma(unsigned int const segment_start,
                           unsigned int const segment_end,
                           arma::colvec const& theta);
  arma::mat GetHessianBinomial(unsigned int const segment_start,
                               unsigned int const segment_end,
                               arma::colvec const& theta);
  arma::mat GetHessianCustom(unsigned int const segment_start,
                             unsigned int const segment_end,
                             arma::colvec const& theta);
  arma::mat GetHessianLm(unsigned int const segment_start,
                         unsigned int const segment_end,
                         arma::colvec const& theta);
  arma::mat GetHessianMa(unsigned int const segment_start,
                         unsigned int const segment_end,
                         arma::colvec const& theta);
  arma::mat GetHessianPoisson(unsigned int const segment_start,
                              unsigned int const segment_end,
                              arma::colvec const& theta);
  void GetNllPeltMean(unsigned int const segment_start,
                      unsigned int const segment_end, bool const cv,
                      std::optional<arma::colvec> const& start);
  void GetNllPeltMean1dValue(unsigned int const segment_start,
                             unsigned int const segment_end, bool const cv,
                             std::optional<arma::colvec> const& start);
  void GetNllPeltMeanXdValue(unsigned int const segment_start,
                             unsigned int const segment_end, bool const cv,
                             std::optional<arma::colvec> const& start);
  void GetNllPeltMeanvariance(unsigned int const segment_start,
                              unsigned int const segment_end, bool const cv,
                              std::optional<arma::colvec> const& start);
  void GetNllPeltMeanvarianceValue(unsigned int const segment_start,
                                   unsigned int const segment_end,
                                   bool const cv,
                                   std::optional<arma::colvec> const& start);
  void GetNllPeltMgaussian(unsigned int const segment_start,
                           unsigned int const segment_end, bool const cv,
                           std::optional<arma::colvec> const& start);
  void GetNllPeltVariance(unsigned int const segment_start,
                          unsigned int const segment_end, bool const cv,
                          std::optional<arma::colvec> const& start);
  void GetNllPeltVarianceValue(unsigned int const segment_start,
                               unsigned int const segment_end, bool const cv,
                               std::optional<arma::colvec> const& start);
  double GetNllSenArma(unsigned int const segment_start,
                       unsigned int const segment_end,
                       arma::colvec const& theta);
  double GetNllSenBinomial(unsigned int const segment_start,
                           unsigned int const segment_end,
                           arma::colvec const& theta);
  double GetNllSenCustom(unsigned int const segment_start,
                         unsigned int const segment_end,
                         arma::colvec const& theta);
  double GetNllSenLasso(unsigned int const segment_start,
                        unsigned int const segment_end,
                        arma::colvec const& theta);
  double GetNllSenLm(unsigned int const segment_start,
                     unsigned int const segment_end, arma::colvec const& theta);
  double GetNllSenMa(unsigned int const segment_start,
                     unsigned int const segment_end, arma::colvec const& theta);
  double GetNllSenPoisson(unsigned int const segment_start,
                          unsigned int const segment_end,
                          arma::colvec const& theta);
  arma::colvec UpdateChangePointSet();
  void UpdateSenParameters();
  void UpdateSenParametersStep(int const segment_start, int const segment_end,
                               int const i);
  void UpdateSenParametersSteps(int const segment_start,
                                unsigned int const segment_end, int const i);
  void UpdateStep();

  arma::colvec active_coefficients_count_;
  double beta_;
  arma::colvec change_points_;
  arma::mat coefficients_;
  arma::mat coefficients_sum_;
  std::string const cost_adjustment_;
  bool const cp_only_;
  arma::mat const data_;
  arma::mat const data_c_;
  unsigned int const data_c_n_rows_;
  double const* data_c_ptr_;
  unsigned int const data_n_dims_;
  unsigned int const data_n_rows_;
  double const epsilon_in_hessian_;
  arma::colvec error_standard_deviation_;
  std::string const family_;
  static std::unordered_map<std::string, FunctionSet> const
      family_function_map_;
  arma::colvec (Fastcpd::* const get_gradient_)(
      unsigned int const segment_start, unsigned int const segment_end,
      arma::colvec const& theta);
  arma::mat (Fastcpd::* const get_hessian_)(unsigned int const segment_start,
                                            unsigned int const segment_end,
                                            arma::colvec const& theta);
  void (Fastcpd::* const get_nll_pelt_)(
      unsigned int const segment_start, unsigned int const segment_end,
      bool const cv, std::optional<arma::colvec> const& start);
  void (Fastcpd::* const get_nll_pelt_value_)(
      unsigned int const segment_start, unsigned int const segment_end,
      bool const cv, std::optional<arma::colvec> const& start);
  double (Fastcpd::* const get_nll_sen_)(unsigned int const segment_start,
                                         unsigned int const segment_end,
                                         arma::colvec const& theta);
  arma::cube hessian_;
  double lasso_penalty_base_;
  arma::colvec line_search_;
  arma::colvec momentum_;
  double const momentum_coef_;
  arma::colvec objective_function_values_;
  arma::colvec objective_function_values_candidates_;
  double* objective_function_values_candidates_ptr_;
  double objective_function_values_min_;
  unsigned int objective_function_values_min_index_;
  arma::colvec const order_;
  unsigned int const parameters_count_;
  arma::colvec const parameters_lower_bound_;
  arma::colvec const parameters_upper_bound_;
  arma::ucolvec pruned_left_;
  unsigned int pruned_left_n_elem_;
  arma::ucolvec pruned_set_;
  unsigned int pruned_set_size_ = 2;
  double const pruning_coefficient_;
  unsigned int const regression_response_count_;
  arma::colvec result_coefficients_;
  arma::mat result_residuals_;
  double result_value_;
  arma::mat segment_coefficients_;
  int const segment_count_;
  arma::colvec segment_indices_;
  unsigned int t = 1;
  double const trim_;
  bool const use_warm_start_;
  double const vanilla_percentage_;
  arma::mat const variance_estimate_;
  arma::mat warm_start_;
};

}  // namespace classes

}  // namespace fastcpd

#endif  // FASTCPD_CLASS_H_
