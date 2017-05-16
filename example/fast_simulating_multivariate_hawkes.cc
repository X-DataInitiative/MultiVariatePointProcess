#include <iostream>
#include <iomanip>
#include <chrono>
#include <Eigen/Dense>
#include "Sequence.h"
#include "PlainHawkes.h"

int main(const int argc, const char** argv)
{
  auto end_time = std::vector<double>();
  end_time.push_back(10000000.0);

  unsigned dim = 2, num_params = dim * (dim + 1);
  Eigen::VectorXd params(num_params);
  params << 0.12, 0.07, 0.6, 0., 1.2, 0.42;

  Eigen::MatrixXd beta(dim,dim);
  beta << 2., 2., 2., 2.;

  PlainHawkes hawkes(num_params, dim, beta);
  hawkes.SetParameters(params);
  std::vector<Sequence> sequences;

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  hawkes.Simulate(end_time, sequences);
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

  std::printf("Simulating %lu events costs %f" ,
              sequences[0].GetEvents().size(), duration / 1000000.);
}
