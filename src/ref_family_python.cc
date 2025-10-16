// Pure C++ implementation of GLM family functions (replacing R-specific ref_r_family.c)
// This provides logit link functions and binomial deviance residuals without R dependencies

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>
#include <string>

namespace {

constexpr double THRESH = 30.0;
constexpr double MTHRESH = -30.0;
constexpr double INVEPS = 1.0 / std::numeric_limits<double>::epsilon();
constexpr double DBL_EPSILON_VAL = std::numeric_limits<double>::epsilon();

// Evaluate x/(1 - x) with bounds checking
inline double x_d_omx(double x) {
    if (x < 0.0 || x > 1.0) {
        throw std::domain_error("Value out of range (0, 1): " + std::to_string(x));
    }
    return x / (1.0 - x);
}

// Evaluate x/(1 + x)
inline double x_d_opx(double x) {
    return x / (1.0 + x);
}

// Helper for binomial deviance: y * log(y/mu)
inline double y_log_y(double y, double mu) {
    return (y != 0.0) ? (y * std::log(y / mu)) : 0.0;
}

}  // namespace

extern "C" {

// Logit link: log(mu/(1-mu))
void logit_link_cpp(const double* mu, double* result, int n) {
    for (int i = 0; i < n; ++i) {
        result[i] = std::log(x_d_omx(mu[i]));
    }
}

// Inverse logit link: exp(eta)/(1+exp(eta))
void logit_linkinv_cpp(const double* eta, double* result, int n) {
    for (int i = 0; i < n; ++i) {
        double etai = eta[i];
        double tmp;
        if (etai < MTHRESH) {
            tmp = DBL_EPSILON_VAL;
        } else if (etai > THRESH) {
            tmp = INVEPS;
        } else {
            tmp = std::exp(etai);
        }
        result[i] = x_d_opx(tmp);
    }
}

// Derivative of inverse logit: d(mu)/d(eta) = exp(eta)/(1+exp(eta))^2
void logit_mu_eta_cpp(const double* eta, double* result, int n) {
    for (int i = 0; i < n; ++i) {
        double etai = eta[i];
        if (etai > THRESH || etai < MTHRESH) {
            result[i] = DBL_EPSILON_VAL;
        } else {
            double expE = std::exp(etai);
            double opexp = 1.0 + expE;
            result[i] = expE / (opexp * opexp);
        }
    }
}

// Binomial deviance residuals: 2*wt*(y*log(y/mu) + (1-y)*log((1-y)/(1-mu)))
void binomial_dev_resids_cpp(const double* y, const double* mu, const double* wt,
                              double* result, int n, int lmu, int lwt) {
    if (lmu == 1) {
        // Single mu value
        double mui = mu[0];
        for (int i = 0; i < n; ++i) {
            double yi = y[i];
            double wti = (lwt > 1) ? wt[i] : wt[0];
            result[i] = 2.0 * wti * (y_log_y(yi, mui) + y_log_y(1.0 - yi, 1.0 - mui));
        }
    } else {
        // Vector of mu values
        for (int i = 0; i < n; ++i) {
            double mui = mu[i];
            double yi = y[i];
            double wti = (lwt > 1) ? wt[i] : wt[0];
            result[i] = 2.0 * wti * (y_log_y(yi, mui) + y_log_y(1.0 - yi, 1.0 - mui));
        }
    }
}

}  // extern "C"
