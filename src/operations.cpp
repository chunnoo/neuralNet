#include "operations.hpp"

Eigen::MatrixXf Operations::relu(const Eigen::MatrixXf &m) {
  return m.unaryExpr([](float e){return e < 0 ? 0 : e;});
}

Eigen::MatrixXf Operations::reluInvDeriv(const Eigen::MatrixXf &m) {
  return m.unaryExpr([](float e){return e <= 0.0f ? 0.0f : 1.0f;});
}

Eigen::MatrixXf Operations::sigmoid(const Eigen::MatrixXf &m) {
  return m.unaryExpr([](float e){return 1/(1 + static_cast<float>(exp(-e)));});
}

Eigen::MatrixXf Operations::sigmoidInvDeriv(const Eigen::MatrixXf &m) {
  return m.unaryExpr([](float e){return e*(1.0f - e);});
}

Eigen::MatrixXf Operations::softmax(const Eigen::MatrixXf &m) {
  /*
  softmax(X, i) = exp(Xi)/sum(exp(X))

  log(sum(exp(X))) = m + log(sum(exp(X - m)))

  log(softmax(X, i)) = Xi - m - log(sum(exp(X - m)))
  */

  const Eigen::RowVectorXf cwm = m.colwise().maxCoeff();

  const Eigen::RowVectorXf esm = (m.rowwise() - cwm).unaryExpr([](float e){return std::exp(e);}).colwise().sum();

  const Eigen::MatrixXf ret = (m.rowwise() - (cwm + esm.unaryExpr([](float e){return std::log(e);}))).unaryExpr([](float e){return std::exp(e);});

  //std::cout << sample(m, 5, 5) << "\n";

  return ret;
}

Eigen::MatrixXf Operations::softmaxInvDeriv(const Eigen::MatrixXf &m) {
  //I think the old definition of this was wrong
  //TODO
  return Eigen::MatrixXf(m.rows(), m.cols());
}

/*Eigen::MatrixXf Operations::meanSquared(const Eigen::MatrixXf &m) {
  //TODO
}

Eigen::MatrixXf Operations::meanSquaredDeriv(const Eigen::MatrixXf &m) {
  //TODO
}

Eigen::MatrixXf Operations::crossEntropy(const Eigen::MatrixXf &m) {
  //TODO
}

Eigen::MatrixXf Operations::crossEntropyDeriv(const Eigen::MatrixXf &m) {
  //TODO
}*/

Eigen::MatrixXf Operations::sample(const Eigen::MatrixXf &m, unsigned int rows, unsigned int cols) {
  Eigen::MatrixXf s(rows, cols);

  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      s(i, j) = m(i*(m.rows()/rows), j*(m.cols()/cols));
    }
  }

  return s;
}
