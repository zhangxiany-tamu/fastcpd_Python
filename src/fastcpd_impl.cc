// Python-compatible implementation (replacing Rcpp version)
#include "fastcpd.h"
#include <functional>

namespace fastcpd {
namespace python {

// Wrapper for Python that returns result components separately
struct FastcpdResult {
    std::vector<double> raw_cp_set;
    std::vector<double> cp_set;
    std::vector<double> cost_values;
    std::vector<std::vector<double>> residuals;
    std::vector<std::vector<double>> thetas;
};

// Main entry point for Python (callable from nanobind)
FastcpdResult fastcpd_impl(
    const arma::mat& data,
    double beta,
    const std::string& cost_adjustment,
    int segment_count,
    double trim,
    double momentum_coef,
    std::function<unsigned int(unsigned int)> multiple_epochs_function,
    const std::string& family,
    double epsilon,
    int p,
    const arma::colvec& order,
    std::function<double(arma::mat)> cost_pelt,
    std::function<double(arma::mat, arma::colvec)> cost_sen,
    std::function<arma::colvec(arma::mat, arma::colvec)> cost_gradient,
    std::function<arma::mat(arma::mat, arma::colvec)> cost_hessian,
    bool cp_only,
    double vanilla_percentage,
    bool warm_start,
    const arma::colvec& lower,
    const arma::colvec& upper,
    const arma::colvec& line_search,
    const arma::mat& variance_estimate,
    unsigned int p_response,
    double pruning_coef,
    bool r_progress = false) {

    // Create Fastcpd instance (without R-specific cost parameter)
    fastcpd::classes::Fastcpd fastcpd_class(
        beta,
#ifndef NO_RCPP
        std::nullopt,  // cost (R function)
#endif
        cost_pelt,
        cost_sen,
        cost_adjustment,
        cost_gradient,
        cost_hessian,
        cp_only,
        data,
        epsilon,
        family,
        multiple_epochs_function,
        line_search,
        lower,
        momentum_coef,
        order,
        p,
        p_response,
        pruning_coef,
        r_progress,
        segment_count,
        trim,
        upper,
        vanilla_percentage,
        variance_estimate,
        warm_start
    );

    // Run the algorithm
    auto result = fastcpd_class.Run();

    // Convert results to Python-friendly format
    FastcpdResult py_result;

    // Extract raw_cp_set
    const arma::colvec& raw_cp = std::get<0>(result);
    py_result.raw_cp_set = std::vector<double>(raw_cp.begin(), raw_cp.end());

    // Extract cp_set
    const arma::colvec& cp = std::get<1>(result);
    py_result.cp_set = std::vector<double>(cp.begin(), cp.end());

    // Extract cost_values
    const arma::colvec& costs = std::get<2>(result);
    py_result.cost_values = std::vector<double>(costs.begin(), costs.end());

    // Extract residuals (matrix -> vector of vectors)
    const arma::mat& residuals = std::get<3>(result);
    py_result.residuals.resize(residuals.n_rows);
    for (size_t i = 0; i < residuals.n_rows; ++i) {
        py_result.residuals[i] = std::vector<double>(
            residuals.row(i).begin(), residuals.row(i).end()
        );
    }

    // Extract thetas (matrix -> vector of vectors)
    const arma::mat& thetas = std::get<4>(result);
    py_result.thetas.resize(thetas.n_rows);
    for (size_t i = 0; i < thetas.n_rows; ++i) {
        py_result.thetas[i] = std::vector<double>(
            thetas.row(i).begin(), thetas.row(i).end()
        );
    }

    return py_result;
}

}  // namespace python
}  // namespace fastcpd
