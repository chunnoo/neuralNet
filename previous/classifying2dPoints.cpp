#include <iostream>
#include <random>
#include <ctime>
#include <algorithm>
#include <png++/png.hpp>
#include "matrix.hpp"
#include "neuralNet.hpp"

std::mt19937 Matrix::rng(static_cast<unsigned int>(time(0)));

int main() {

  const unsigned int batchSize = 16;
  const unsigned int trainingBatches = 16;
  const float alpha = 0.001f;
  const float dropoutRate = 0.0f;

  std::vector<Matrix> xs;
  std::vector<Matrix> ys;

  unsigned int imgWidth = 128;
  unsigned int imgHeight = 128;

  unsigned int testSize = 1024;
  Matrix testX(2, testSize);
  testX.randomFill(-10.0, 10.0);

  png::image<png::rgb_pixel> imgX(imgWidth,imgHeight);
  png::image<png::rgb_pixel> imgY(imgWidth,imgHeight);

  //std::vector<unsigned int> layerSizes{2, 4, 4, 2};
  //std::vector<Activation> layerActivations{NONE, RELU, RELU, SIGMOID};
  NeuralNet nn({2, 4, 4, 2}, {NONE, RELU, RELU, SIGMOID});

  //create training batches
  for (unsigned int i = 0; i < trainingBatches; i++) {
    Matrix x(2, batchSize);
    Matrix y(2, batchSize);
    x.randomFill(-10.0, 10.0);
    for (unsigned int j = 0; j < batchSize; j++) {
      if (x.get(0, j)*x.get(0, j) + x.get(1, j)*x.get(1, j) <= 25) {
        y.set(0, j, 1.0);
        y.set(1, j, 0.0);
      } else {
        y.set(0, j, 0.0);
        y.set(1, j, 1.0);
      }
    }
    xs.push_back(x);
    ys.push_back(y);
  }

  nn.backPropagation(xs, ys, alpha, dropoutRate, 1024*64, 1024);

  Matrix testY = nn.use(testX);

  //draw image of training batches
  for (unsigned int i = 0; i < trainingBatches; i++) {
    for (unsigned int j = 0; j < batchSize; j++) {
      imgX[static_cast<unsigned int>((xs[i].get(0, j)*0.1f + 1)*static_cast<float>(imgWidth)*0.5f)][static_cast<unsigned int>((xs[i].get(1, j)*0.1f + 1)*static_cast<float>(imgHeight)*0.5f)] = png::rgb_pixel(static_cast<unsigned char>(ys[i].get(0,j)*255), static_cast<unsigned char>(ys[i].get(1,j)*255), 0);
    }
  }
  imgX.write("images/classifying2dPointsTraining.png");

  //draw image of test batch
  for (unsigned int i = 0; i < testSize; i++) {
    imgY[static_cast<unsigned int>((testX.get(0, i)*0.1f + 1)*static_cast<float>(imgWidth)*0.5f)][static_cast<unsigned int>((testX.get(1, i)*0.1f + 1)*static_cast<float>(imgHeight)*0.5f)] = png::rgb_pixel(static_cast<unsigned char>(testY.get(0,i)*255), static_cast<unsigned char>(testY.get(1,i)*255), 0);
  }
  imgY.write("images/classifying2dPointsTest.png");

  return 0;
}
