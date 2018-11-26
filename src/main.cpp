#include <iostream>
#include <random>
#include <ctime>
#include <algorithm>
#include <string>
#include <png++/png.hpp>
#include "matrix.hpp"
#include "mnist.hpp"
#include "neuralNet.hpp"

std::mt19937 Matrix::rng(static_cast<unsigned int>(time(0)));

int main() {

  Mnist trainData("train");
  Mnist testData("t10k");

  const unsigned int batchSize = 512;

  std::vector<Matrix> trainLabelBatches = trainData.getLabelBatches(batchSize);
  std::vector<Matrix> trainImageBatches = trainData.getImageBatches(batchSize);

  std::vector<Matrix> testLabelBatches = testData.getLabelBatches(testData.getNumLabels());
  std::vector<Matrix> testImageBatches = testData.getImageBatches(testData.getNumImages());

  if (trainLabelBatches.size() != trainImageBatches.size() || testLabelBatches.size() != testImageBatches.size()) {
    std::cout << "fuck" << std::endl;
  }

  const float alpha = 0.00001f;
  const float dropoutRate = 0.2f;

  NeuralNet nn({trainData.getImageHeight()*trainData.getImageWidth(), 256, 128, 128, 10}, {NONE, RELU, RELU, RELU, SIGMOID});
  //NeuralNet nn("test");

  nn.backPropagation(trainImageBatches, trainLabelBatches, alpha, dropoutRate, 1024, 16);

  Matrix testOutput = nn.use(testImageBatches[0]);

  std::cout << testOutput.getSample(10, 16) << std::endl;
  std::cout << testLabelBatches[0].getSample(10, 16) << std::endl;

  float testError = testOutput.subtract(testLabelBatches[0]).absAvg();

  std::cout << testError << std::endl;

  nn.save("test");

  /*std::cout << labelBatches[1] << std::endl;

  unsigned int imgHeight = trainMnist.getImageHeight();
  unsigned int imgWidth = trainMnist.getImageWidth();
  png::image<png::gray_pixel> mnistImage(imgWidth, imgHeight);

  for (unsigned int i = 0; i < imgWidth; i++) {
    for (unsigned int j = 0; j < imgHeight; j++) {
      mnistImage[i][j] = png::gray_pixel(imageBatches[1].get(imgWidth*i + j, 0)*255);
    }
  }

  mnistImage.write("images/mnistImage.png");*/

  /*

  std::vector<Matrix> xs;
  std::vector<Matrix> ys;

  unsigned int imgWidth = 128;
  unsigned int imgHeight = 128;

  unsigned int testSize = 1024;
  Matrix testX(2, testSize);
  testX.randomFill(-10.0, 10.0);

  png::image<png::rgb_pixel> imgX(imgWidth,imgHeight);
  png::image<png::rgb_pixel> imgY(imgWidth,imgHeight);

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



  nn.printWeightsAndBiases();

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
  imgY.write("images/classifying2dPointsTest.png");*/

  return 0;
}
