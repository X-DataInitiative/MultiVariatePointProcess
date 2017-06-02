#include <iostream>
#include <iomanip>
#include <chrono>
#include <Eigen/Dense>
#include <array>
#include "Sequence.h"
#include "PlainHawkes.h"
#include "tick_hawkes_data.h"

int main(const int argc, const char **argv) {

  unsigned int n_simulations = 100;
  std::array<unsigned int, 3> n_nodes_sample = {{1, 2, 4, 16}};
  std::array<double, 7> end_times = {{1000, 2000, 5000, 10000, 20000, 50000, 100000}};

  for (auto n_nodes : n_nodes_sample) {
    for (auto end_time_value : end_times) {
      auto end_time = std::vector<double>();
      for (int i = 0; i < n_simulations; ++i) {
        end_time.push_back(end_time_value);
      }

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

      unsigned long n_events = 0;
      for (int i = 0; i < n_simulations; ++i) {
        n_events += sequences[i].GetEvents().size();
      }

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
