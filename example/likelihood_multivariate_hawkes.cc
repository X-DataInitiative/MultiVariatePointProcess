#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "Sequence.h"
#include "PlainHawkes.h"

const std::string PATH_TO_TICK = "../../../../tick/";
const std::string SIMULATION_FILE = "benchmarks/hawkes_data/hawkes_simulation_by_node.txt";
const std::string TEST_COEFFS_FILE = "benchmarks/hawkes_data/hawkes_test_coeffs.txt";

int main(const int argc, const char **argv) {
  double decay = 2.;

  // Extract Hawkes realization
  std::ifstream ifile(PATH_TO_TICK + SIMULATION_FILE, std::ios::in);

  double num = 0.0;
  unsigned int node = 0;

  unsigned int eventID = 0;
  std::vector<Event> events;
  std::string line;

  while (std::getline(ifile, line)) {
    std::istringstream sin(line);
    while (sin >> num) {
      Event event;
      event.EventID = eventID;
      ++eventID;
      event.SequenceID = 0;
      event.DimentionID = node;
      event.time = num;
      event.marker = 0;
      events.push_back(event);
    }
    ++node;
  }

  const unsigned int n_nodes = node;

  // Construct Sequence of sorted events
  struct sort_pred {
    bool operator()(const Event &left, const Event &right) {
      return left.time < right.time;
    }
  };
  std::sort(events.begin(), events.end(), sort_pred());
  std::vector<Sequence> sequences(1);
  sequences[0] = Sequence(100);
  for (auto event : events) {
    sequences[0].Add(event);
  }

  // Extract test coefficients
  unsigned num_params = n_nodes * (n_nodes + 1);
  std::ifstream test_coeffs_file(PATH_TO_TICK + TEST_COEFFS_FILE, std::ios::in);

  std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > test_coeffs;

  unsigned int line_number = 0;
  while (std::getline(test_coeffs_file, line)) {
    std::istringstream sin(line);
    test_coeffs.push_back(Eigen::VectorXd(num_params));
    for (int j = 0; j < num_params && sin >> num; ++j) {
      if (j > n_nodes) num *= decay;
      test_coeffs[line_number](j) = num;
    }
    ++line_number;
  }

  // Create Hawkes model
  Eigen::VectorXd params(num_params);

  Eigen::MatrixXd beta(n_nodes, n_nodes);
  beta << decay, decay, decay, decay;

  PlainHawkes hawkes(num_params, n_nodes, beta);

  // Precompute weights
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  unsigned n_tries = 5;
  for (int i = 0; i < n_tries; ++i) {
    PlainHawkes::OPTION options;
    options.ini_max_iter = 0;
    options.method = PlainHawkes::SGD;
    hawkes.fit(sequences, options);
  }
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Time needed to compute weights " << duration / (1000000.0 * n_tries)
            << " secs." << std::endl;

  // Compute log likelihood
  double llh = 0;
  Eigen::VectorXd gradient(num_params);

  t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < line_number; ++i) {
    hawkes.SetParameters(test_coeffs[0]);
    hawkes.NegLoglikelihood(llh, gradient);
  }
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Average time to compute likelihood " << duration / (1000000.0 * line_number)
            << " secs." << std::endl;

  std::printf("Negative loglikelihood value on first test coeff %.6f", llh);
}
