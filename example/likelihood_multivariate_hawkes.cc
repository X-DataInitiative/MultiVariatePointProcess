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
  std::array<double, 7> end_times = {{10000., 20000, 50000, 100000, 200000, 500000, 1000000}};

  for (auto end_time : end_times) {
    for (auto n_nodes : n_nodes_sample) {

      Eigen::MatrixXd beta = get_beta(n_nodes);

      auto sequences = get_simulation(n_nodes, end_time);
      int n_events = sequences[0].GetEvents().size();
      auto test_coeffs = import_test_coeffs(n_nodes, beta(0, 0));

      // Create Hawkes model
      unsigned num_params = n_nodes * (n_nodes + 1);
      Eigen::VectorXd params(num_params);

      PlainHawkes hawkes(num_params, n_nodes, beta);
      double llh = 0;
      Eigen::VectorXd gradient(num_params);

      // compute first likelihood
      std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
      unsigned n_tries = 1;
      for (int i = 0; i < n_tries; ++i) {
        PlainHawkes::OPTION options;
        options.ini_max_iter = 0;
        options.method = PlainHawkes::SGD;
        hawkes.fit(sequences, options);
        hawkes.SetParameters(test_coeffs[0]);
        hawkes.NegLoglikelihood(llh, gradient);
      }
      std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
      double first_likelihood_time =
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
              / (1000000.0 * n_tries);

      // Compute log likelihood
      t1 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < test_coeffs.size(); ++i) {
        hawkes.SetParameters(test_coeffs[i]);
        hawkes.NegLoglikelihood(llh, gradient);
      }
      t2 = std::chrono::high_resolution_clock::now();
      double average_compute_likelihood_time =
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
              / (1000000.0 * test_coeffs.size());

      if (end_time == 100) {
        std::cout << "Time needed to compute weights " << first_likelihood_time
                  << " secs." << std::endl;

        std::cout << "Average time to compute likelihood "
                  << average_compute_likelihood_time
                  << " secs." << std::endl;

        std::printf("Negative loglikelihood value on first test coeff %.6f", llh);
      } else {
        std::printf("likelihood,PtPack,%i,%i,%.6f\n",
                    n_nodes,
                    n_events,
                    average_compute_likelihood_time);
        std::printf("first likelihood,PtPack,%i,%i,%.6f\n",
                    n_nodes, n_events, first_likelihood_time);

      }
    }
  }
}
