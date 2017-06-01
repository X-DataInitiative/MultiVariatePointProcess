#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <array>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "Sequence.h"
#include "PlainHawkes.h"

#ifndef MULTIVARIATEPOINTPROCESS_TICK_HAWKES_DATA_H
#define MULTIVARIATEPOINTPROCESS_TICK_HAWKES_DATA_H

const std::string PATH_TO_TICK = "../../../../tick/";
const std::string
    SIMULATION_FILE = "benchmarks/hawkes_data/hawkes_simulation_n_nodes_%i_end_time_%i.txt";
const std::string TEST_COEFFS_FILE = "benchmarks/hawkes_data/hawkes_test_coeffs_n_nodes_%i.txt";
const std::string COEFFS_FILE = "benchmarks/hawkes_data/hawkes_coeffs_n_nodes_%i.txt";

extern std::vector<Sequence> get_simulation(unsigned int n_nodes, int end_time) {
  char full_path_buffer[150];
  std::sprintf(full_path_buffer, "%s%s", PATH_TO_TICK.c_str(), SIMULATION_FILE.c_str());
  char filename_buffer[150];
  std::sprintf(filename_buffer, full_path_buffer, n_nodes, end_time);
  std::ifstream ifile(filename_buffer, std::ios::in);

  double num = 0.0;
  unsigned int node = 0;

  unsigned int eventID = 0;
  std::vector<Event> events;
  std::string line;

  while (std::getline(ifile, line)) {
    std::istringstream line_stream(line);
    while (line_stream >> num) {
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

  // Construct Sequence of sorted events
  struct sort_pred {
    bool operator()(const Event &left, const Event &right) {
      return left.time < right.time;
    }
  };
  std::sort(events.begin(), events.end(), sort_pred());
  std::vector<Sequence> sequences(1);
  sequences[0] = Sequence();
  for (auto event : events) {
    sequences[0].Add(event);
  }
  return sequences;
}

Eigen::MatrixXd get_beta(unsigned int n_nodes) {
  Eigen::MatrixXd beta(n_nodes, n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    for (int j = 0; j < n_nodes; ++j) {
      if (n_nodes == 1) {
        beta(i, j) = 1.5;
      } else if (n_nodes == 2) {
        beta(i, j) = 2.;
      } else if (n_nodes == 4) {
        beta(i, j) = 0.5;
      }
    }
  }
  return beta;
}

extern Eigen::VectorXd get_parameters(unsigned int n_nodes){
  auto beta = get_beta(n_nodes);
  // beta is a constant matrix
  double beta_mean = beta.mean();

  char coeff_full_path_buffer[150];
  std::sprintf(coeff_full_path_buffer, "%s%s", PATH_TO_TICK.c_str(), COEFFS_FILE.c_str());

  char coeff_file_buffer[150];
  std::sprintf(coeff_file_buffer, coeff_full_path_buffer, n_nodes);

  std::ifstream test_coeffs_file(coeff_file_buffer, std::ios::in);

  unsigned num_params = n_nodes * (n_nodes + 1);
  Eigen::VectorXd coeffs(num_params);

  unsigned int line_number = 0;
  std::string line;
  while (std::getline(test_coeffs_file, line)) {
    std::istringstream line_stream(line);
    double num = 0.0;
    for (int j = 0; j < n_nodes && line_stream >> num; ++j) {
      // we are filling baseline
      if (line_number == 0) {
        coeffs(line_number * n_nodes + j) = num;
      }
      else {
        // we are filling alpha
        num *= beta_mean;
        coeffs(line_number * n_nodes + j) = num;
      }
    }
    ++line_number;
  }

  return coeffs;
}
//
//extern std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> get_parameters(unsigned int n_nodes){
//  auto beta = get_beta(n_nodes);
//  // beta is a constant matrix
//  double beta_mean = beta.mean();
//
//  char coeff_full_path_buffer[150];
//  std::sprintf(coeff_full_path_buffer, "%s%s", PATH_TO_TICK.c_str(), COEFFS_FILE.c_str());
//
//  char coeff_file_buffer[150];
//  std::sprintf(coeff_file_buffer, coeff_full_path_buffer, n_nodes);
//
//  std::ifstream test_coeffs_file(coeff_file_buffer, std::ios::in);
//
//  Eigen::VectorXd baseline(n_nodes);
//  Eigen::MatrixXd alpha(n_nodes, n_nodes);
//
//  unsigned int line_number = 0;
//  std::string line;
//  while (std::getline(test_coeffs_file, line)) {
//    std::istringstream line_stream(line);
//    double num = 0.0;
//    for (int j = 0; j < n_nodes && line_stream >> num; ++j) {
//      // we are filling baseline
//      if (line_number == 0) {
//        baseline(j) = num;
//      }
//      else {
//        // we are filling alpha
//        if (j > n_nodes) num *= beta_mean;
//        alpha(line_number - 1, j) = num;
//      }
//    }
//    ++line_number;
//  }
//  return std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>(beta, baseline, alpha);
//
//};

extern std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> >
import_test_coeffs(unsigned int n_nodes, double decay) {
  char coeff_full_path_buffer[150];
  std::sprintf(coeff_full_path_buffer, "%s%s", PATH_TO_TICK.c_str(), TEST_COEFFS_FILE.c_str());

  char coeff_file_buffer[150];
  std::sprintf(coeff_file_buffer, coeff_full_path_buffer, n_nodes);

  unsigned num_params = n_nodes * (n_nodes + 1);
  std::ifstream test_coeffs_file(coeff_file_buffer, std::ios::in);

  std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > test_coeffs;

  unsigned int line_number = 0;
  std::string line;
  while (std::getline(test_coeffs_file, line)) {
    std::istringstream line_stream(line);
    double num = 0.0;
    test_coeffs.push_back(Eigen::VectorXd(num_params));
    for (int j = 0; j < num_params && line_stream >> num; ++j) {
      if (j > n_nodes) num *= decay;
      test_coeffs[line_number](j) = num;
    }
    ++line_number;
  }
  return test_coeffs;
};

#endif //MULTIVARIATEPOINTPROCESS_TICK_HAWKES_DATA_H
