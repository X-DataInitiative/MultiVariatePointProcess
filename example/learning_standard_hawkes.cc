#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <array>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "Sequence.h"
#include "PlainHawkes.h"
#include "tick_hawkes_data.h"

int main(const int argc, const char **argv) {
  std::array<unsigned int, 4> n_nodes_sample = {{1, 2, 4, 16}};
//  std::array<double, 1> end_times = {{100}}
  std::array<double, 1> end_times = {{10000}};//10000., 20000}};//, 50000, 100000, 200000, 500000, 1000000}};

  for (auto end_time : end_times) {
    for (auto n_nodes : n_nodes_sample) {
      unsigned num_params = n_nodes * (n_nodes + 1);

      auto sequences = get_simulation(n_nodes, end_time);
      int n_events = sequences[0].GetEvents().size();

      Eigen::MatrixXd beta = get_beta(n_nodes);

      auto t1 = std::chrono::high_resolution_clock::now();
      PlainHawkes hawkes_new(num_params, n_nodes, beta);
      PlainHawkes::OPTION options;
      options.method = PlainHawkes::PLBFGS;
      options.base_intensity_regularizer = PlainHawkes::NONE;
      options.excitation_regularizer = PlainHawkes::NONE;
      options.ini_max_iter = 1000;
      options.verbose = end_time == 100;

      hawkes_new.fit(sequences, options);
      auto t2 = std::chrono::high_resolution_clock::now();
      double fit_time =
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
              / (1000000.0);

      auto diff = hawkes_new.GetParameters() - get_parameters(n_nodes);
      auto norm_diff = diff.norm() / diff.size();

      if (end_time == 100) {
        std::printf("It has taken %.6f secs\n", fit_time);

        std::cout << "Estimated Parameters : " << std::endl;
        std::cout << hawkes_new.GetParameters().transpose() << std::endl;
        std::cout << "True Parameters : " << std::endl;
        std::cout << get_parameters(n_nodes).transpose() << std::endl;

        std::cout << "norm_diff " << norm_diff << std::endl;
      }
      else {
        if (norm_diff < 0.1) {
          std::printf("fit,PtPack,%i,%i,%.6f", n_nodes, n_events, fit_time);
        }
        else {
          std::printf("fit,PtPack,%i,%i,fail", n_nodes, n_events);
        }
      }
    }
  }
  return 0;
}