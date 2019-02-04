#ifndef INCLUDED_OPERATIONS
#define INCLUDED_OPERATIONS

#include <Eigen/Dense>
#include <iostream>
#include <utility>
#include <cmath>

namespace Operations {

  Eigen::MatrixXf relu(const Eigen::MatrixXf &m);
  Eigen::MatrixXf reluInvDeriv(const Eigen::MatrixXf &m);
  Eigen::MatrixXf sigmoid(const Eigen::MatrixXf &m);
  Eigen::MatrixXf sigmoidInvDeriv(const Eigen::MatrixXf &m);
  Eigen::MatrixXf softmax(const Eigen::MatrixXf &m);
  Eigen::MatrixXf softmaxInvDeriv(const Eigen::MatrixXf &m);

  Eigen::MatrixXf meanSquared(const Eigen::MatrixXf &m);
  Eigen::MatrixXf meanSquaredDeriv(const Eigen::MatrixXf &m);
  Eigen::MatrixXf crossEntropy(const Eigen::MatrixXf &m);
  Eigen::MatrixXf crossEntropyDeriv(const Eigen::MatrixXf &m);

  Eigen::MatrixXf sample(const Eigen::MatrixXf &m, unsigned int rows, unsigned int cols);

}

#endif
