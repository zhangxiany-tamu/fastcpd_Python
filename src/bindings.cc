// nanobind bindings for fastcpd
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>

#include "fastcpd_impl.cc"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_fastcpd_impl, m) {
    m.doc() = "Fast change point detection - C++ implementation";

    // Bind the FastcpdResult structure
    nb::class_<fastcpd::python::FastcpdResult>(m, "FastcpdResult")
        .def_ro("raw_cp_set", &fastcpd::python::FastcpdResult::raw_cp_set,
                "Raw change point set")
        .def_ro("cp_set", &fastcpd::python::FastcpdResult::cp_set,
                "Change point set")
        .def_ro("cost_values", &fastcpd::python::FastcpdResult::cost_values,
                "Cost values for each segment")
        .def_ro("residuals", &fastcpd::python::FastcpdResult::residuals,
                "Residuals")
        .def_ro("thetas", &fastcpd::python::FastcpdResult::thetas,
                "Parameter estimates");

    // Main fastcpd implementation function
    m.def("fastcpd_impl",
        [](nb::ndarray<double, nb::ndim<2>, nb::c_contig> data_np,
           double beta,
           const std::string& cost_adjustment,
           int segment_count,
           double trim,
           double momentum_coef,
           nb::object multiple_epochs_function,  // Accept None or callable
           const std::string& family,
           double epsilon,
           int p,
           const std::vector<double>& order,
           nb::object cost_pelt_py,  // Accept None or callable
           nb::object cost_sen_py,  // Accept None or callable
           nb::object cost_gradient_py,  // Accept None or callable
           nb::object cost_hessian_py,  // Accept None or callable
           bool cp_only,
           double vanilla_percentage,
           bool warm_start,
           const std::vector<double>& lower,
           const std::vector<double>& upper,
           const std::vector<double>& line_search,
           nb::ndarray<double, nb::ndim<2>, nb::c_contig> variance_estimate_np,
           unsigned int p_response,
           double pruning_coef,
           bool r_progress) -> fastcpd::python::FastcpdResult {

            // Convert numpy array to Armadillo matrix
            size_t n_rows = data_np.shape(0);
            size_t n_cols = data_np.shape(1);

            arma::mat data(n_rows, n_cols);
            double* data_ptr = data_np.data();
            for (size_t i = 0; i < n_rows; ++i) {
                for (size_t j = 0; j < n_cols; ++j) {
                    data(i, j) = data_ptr[i * n_cols + j];
                }
            }

            // Convert order vector to Armadillo
            arma::colvec order_arma(order.size());
            for (size_t i = 0; i < order.size(); ++i) {
                order_arma(i) = order[i];
            }

            // Convert variance_estimate numpy array to Armadillo matrix
            arma::mat variance_estimate;
            if (variance_estimate_np.size() > 0) {
                size_t ve_rows = variance_estimate_np.shape(0);
                size_t ve_cols = variance_estimate_np.shape(1);
                variance_estimate.set_size(ve_rows, ve_cols);
                double* ve_ptr = variance_estimate_np.data();
                for (size_t i = 0; i < ve_rows; ++i) {
                    for (size_t j = 0; j < ve_cols; ++j) {
                        variance_estimate(i, j) = ve_ptr[i * ve_cols + j];
                    }
                }
            }

            // Convert bounds vectors to Armadillo
            arma::colvec lower_arma(lower.size());
            for (size_t i = 0; i < lower.size(); ++i) {
                lower_arma(i) = lower[i];
            }

            arma::colvec upper_arma(upper.size());
            for (size_t i = 0; i < upper.size(); ++i) {
                upper_arma(i) = upper[i];
            }

            arma::colvec line_search_arma(line_search.size());
            for (size_t i = 0; i < line_search.size(); ++i) {
                line_search_arma(i) = line_search[i];
            }

            // Wrap Python cost functions to work with Armadillo types
            // For now, we pass nullptr for custom cost functions (built-in families work)
            std::function<double(arma::mat)> cost_pelt = nullptr;
            std::function<double(arma::mat, arma::colvec)> cost_sen = nullptr;
            std::function<arma::colvec(arma::mat, arma::colvec)> cost_gradient = nullptr;
            std::function<arma::mat(arma::mat, arma::colvec)> cost_hessian = nullptr;
            std::function<unsigned int(unsigned int)> epochs_func = nullptr;

            // Call the C++ implementation with error handling
            try {
                return fastcpd::python::fastcpd_impl(
                    data, beta, cost_adjustment, segment_count, trim, momentum_coef,
                    epochs_func, family, epsilon, p, order_arma,
                    cost_pelt, cost_sen, cost_gradient, cost_hessian,
                    cp_only, vanilla_percentage, warm_start,
                    lower_arma, upper_arma, line_search_arma, variance_estimate,
                    p_response, pruning_coef, r_progress
                );
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("fastcpd C++ error: ") + e.what());
            }
        },
        nb::arg("data"),
        nb::arg("beta"),
        nb::arg("cost_adjustment"),
        nb::arg("segment_count"),
        nb::arg("trim"),
        nb::arg("momentum_coef"),
        nb::arg("multiple_epochs_function").none(),  // Allow None
        nb::arg("family"),
        nb::arg("epsilon"),
        nb::arg("p"),
        nb::arg("order"),
        nb::arg("cost_pelt").none(),  // Allow None
        nb::arg("cost_sen").none(),   // Allow None
        nb::arg("cost_gradient").none(),  // Allow None
        nb::arg("cost_hessian").none(),   // Allow None
        nb::arg("cp_only"),
        nb::arg("vanilla_percentage"),
        nb::arg("warm_start"),
        nb::arg("lower"),
        nb::arg("upper"),
        nb::arg("line_search"),
        nb::arg("variance_estimate"),
        nb::arg("p_response"),
        nb::arg("pruning_coef"),
        nb::arg("r_progress") = false,
        "Fast change point detection implementation"
    );
}
