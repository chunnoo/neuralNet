#include <iostream>
#include <random>
#include <ctime>
#include <algorithm>
#include <string>
#include "operations.hpp"
#include <png++/png.hpp>
#include <Eigen/Dense>
#include "mnist.hpp"
#include "neuralNet.hpp"

int main() {

  srand(static_cast<unsigned int>(time(0)));

  Mnist trainData("train");
  Mnist testData("t10k");

  const unsigned int batchSize = 6000;//trainData.getNumLabels();

  std::vector<Eigen::MatrixXf> trainLabelBatches = trainData.getLabelBatches(batchSize, 1);
  std::vector<Eigen::MatrixXf> trainImageBatches = trainData.getImageBatches(batchSize, 1);

  std::vector<Eigen::MatrixXf> testLabelBatches = testData.getLabelBatches(testData.getNumLabels());
  std::vector<Eigen::MatrixXf> testImageBatches = testData.getImageBatches(testData.getNumImages());

  if (trainLabelBatches.size() != trainImageBatches.size() || testLabelBatches.size() != testImageBatches.size()) {
    std::cout << "fuck" << std::endl;
  }

  const float alpha = 0.0001f;
  const float dropoutRate = 0.05f;

  NeuralNet nn({trainData.getImageHeight()*trainData.getImageWidth(), 256, 128, 128, 10}, {NONE, SIGMOID, SIGMOID, SIGMOID, SOFTMAX});
  //NeuralNet nn("firstHalfWoringMnistNetwork");

  while (true) {

    nn.backPropagation(trainImageBatches, trainLabelBatches, CROSSENTROPY, alpha, dropoutRate, 64, 16);

    Eigen::MatrixXf testOutput = nn.use(testImageBatches[0]);

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << Operations::sample(testOutput, 10, 16);
    std::cout << "\n\n";
    std::cout << Operations::sample(testLabelBatches[0], 10, 16);
    std::cout << "\n";
    std::cout.precision(8);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);

    float testError = (testOutput - testLabelBatches[0]).unaryExpr([](float e){return std::abs(e);}).mean();

    std::cout << testError << std::endl;

    nn.save("test");

  }

  return 0;
}
