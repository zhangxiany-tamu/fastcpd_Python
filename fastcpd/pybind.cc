#include "cc/cclib/class.h"

#define PY_SSIZE_T_CLEAN
#define CONFIG_64
#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<double> fastcpd_impl(
  std::vector<std::vector<double>> &data,
  std::vector<std::vector<double>> &variance_estimate
) {
  // Convert input data to arma::mat
  arma::mat data_mat(data.size(), data[0].size());
  for (size_t i = 0; i < data.size(); ++i) {
    for (size_t j = 0; j < data[i].size(); ++j) {
      data_mat(i, j) = data[i][j];
    }
  }

  // Convert variance_estimate to arma::mat
  arma::mat variance_estimate_mat(variance_estimate.size(), variance_estimate[0].size());
  for (size_t i = 0; i < variance_estimate.size(); ++i) {
    for (size_t j = 0; j < variance_estimate[i].size(); ++j) {
      variance_estimate_mat(i, j) = variance_estimate[i][j];
    }
  }

  fastcpd::classes::Fastcpd fastcpd_instance(
      /* beta */ 10.36163,
      /* cost_adjustment */ "MBIC",
      /* cp_only */ false,
      /* data */ data_mat,
      /* epsilon */ 0.0,
      /* family */ "mean",
      /* line_search */ arma::colvec(),
      /* lower */ arma::colvec(),
      /* momentum_coef */ 0.0,
      /* order */ arma::colvec(),
      /* p */ 3,
      /* p_response */ 0,
      /* pruning_coef */ 0.6931472,
      /* segment_count */ 10,
      /* trim */ 0.05,
      /* upper */ arma::colvec(),
      /* vanilla_percentage */ 1.0,
      /* variance_estimate */ variance_estimate_mat,
      /* warm_start */ false);

  std::tuple<arma::colvec, arma::colvec, arma::colvec, arma::mat, arma::mat>
      result = fastcpd_instance.Run();

  arma::colvec change_points = std::get<1>(result);

  return std::vector<double>(change_points.begin(), change_points.end());
}

PYBIND11_MODULE(interface, m)
{
    m.doc() = "fastcpd C++/Python interface";
    m.def("fastcpd_impl", &fastcpd_impl, "A function that computes the fast change point detection");
}
