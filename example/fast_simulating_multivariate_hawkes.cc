#include <iostream>
#include <iomanip>
#include <chrono>
#include <Eigen/Dense>
#include <array>
#include "Sequence.h"
#include "PlainHawkes.h"
#include "tick_hawkes_data.h"

Eigen::VectorXd get_parameters(unsigned int n_nodes) {
  unsigned int num_params = n_nodes * (n_nodes + 1);
  Eigen::VectorXd params(num_params);
  if (n_nodes == 1) {
    params << 0.4, 0.9;
  } else if (n_nodes == 2) {
    params << 0.25, 0.3, 0.6, 0., 1.2, 0.4;
  } else if (n_nodes == 4) {
    params << 0.17, 0., 0.12, 0.09,
        0.15, 0., 0.1, 0.05,
        0.15, 0.1, 0.05, 0.,
        0.05, 0.1, 0., 0.15,
        0.1, 0., 0.05, 0.1;
  }
  return params;
}


int main(const int argc, const char **argv) {

  std::array<unsigned int, 3> n_nodes_sample = {{1, 2, 4}};
  std::array<double, 7> end_times = {{100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000}};

  for (auto n_nodes : n_nodes_sample) {
    for (auto end_time_value : end_times) {
      auto end_time = std::vector<double>();
      end_time.push_back(end_time_value);

      unsigned int num_params = n_nodes * (n_nodes + 1);
      auto params = get_parameters(n_nodes);
      auto beta = get_beta(n_nodes);

      PlainHawkes hawkes(num_params, n_nodes, beta);
      hawkes.SetParameters(params);
      std::vector<Sequence> sequences;

      std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
      hawkes.Simulate(end_time, sequences);
      std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
      auto simulation_time= std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
          / 1000000.;

      unsigned long n_events = sequences[0].GetEvents().size();
      if (false) {
        std::printf("Simulating %lu events costs %f\n",
                   n_events, simulation_time);
      } else {
        std::printf("simulation,PtPack,%i,%lu,%.6f\n",
                    n_nodes, n_events, simulation_time);
      }
    }
  }
}
