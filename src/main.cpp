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
  std::cout.precision(8);
  std::cout.setf(std::ios::fixed, std::ios::floatfield);

  Mnist trainData("train");
  Mnist testData("t10k");

  const unsigned int batchSize = 128;//trainData.getNumLabels();

  std::vector<Eigen::MatrixXf> trainLabelBatches = trainData.getLabelBatches(batchSize);
  std::vector<Eigen::MatrixXf> trainImageBatches = trainData.getImageBatches(batchSize);

  std::vector<Eigen::MatrixXf> testLabelBatches = testData.getLabelBatches(testData.getNumLabels());
  std::vector<Eigen::MatrixXf> testImageBatches = testData.getImageBatches(testData.getNumImages());

  if (trainLabelBatches.size() != trainImageBatches.size() || testLabelBatches.size() != testImageBatches.size()) {
    std::cout << "fuck" << std::endl;
  }

  const float alpha = 0.005f;
  const float dropoutRate = 0.05f;

  //NeuralNet nn({trainData.getImageHeight()*trainData.getImageWidth(), 256, 128, 128, 10}, {NONE, SIGMOID, SIGMOID, SIGMOID, SOFTMAX});
  NeuralNet nn("firstSuccessfulMnistNetwork");

  while (true) {

    nn.backPropagation(trainImageBatches, trainLabelBatches, CROSSENTROPY, alpha, dropoutRate, 512, 16);

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

    nn.save("firstActuallyWorkingMnistNetworkWithDropout3");

  }

  /*unsigned int samples = 8;
  unsigned int outputSize = 10;

  Eigen::MatrixXf testOutput = nn.use(testImageBatches[0]);

  std::vector<unsigned int> sampleIndices = testData.getSortedSampleIndices(samples);

  png::image<png::rgb_pixel> sampleImg(samples * outputSize, 2 * outputSize);

  for (unsigned int i = 0; i < sampleIndices.size(); i++) {
    for (unsigned int j = 0; j < outputSize; j++) {
      sampleImg[j][i] = png::rgb_pixel(testOutput(j, sampleIndices[i]) * 255.0f, 0.0f, 0.0f);
      sampleImg[j + outputSize][i] = png::rgb_pixel(testLabelBatches[0](j, sampleIndices[i]) * 255.0f, 0.0f, 0.0f);
    }
  }

  sampleImg.write("images/mnistOutput.png");

  png::image<png::gray_pixel> sampleInputImg(testData.getImageWidth() * samples, testData.getImageHeight() * outputSize);

  for (unsigned int i = 0; i < sampleIndices.size(); i++) {
    for (unsigned int j = 0; j < testData.getImageHeight(); j++) {
      for (unsigned int k = 0; k < testData.getImageWidth(); k++) {
        sampleInputImg[j + (i / samples)*testData.getImageHeight()][k + (i % samples)*testData.getImageWidth()] = png::gray_pixel(testImageBatches[0](testData.getImageWidth()*j + k, sampleIndices[i])*255.0f);
      }
    }
  }

  std::ofstream file("images/test.txt", std::ios::binary);

  unsigned int sampleInputImgWidth = sampleInputImg.get_width();
  unsigned int sampleInputImgHeight = sampleInputImg.get_height();
  file.write(reinterpret_cast<char *>(&sampleInputImgWidth), sizeof(unsigned int));
  file.write(reinterpret_cast<char *>(&sampleInputImgHeight), sizeof(unsigned int));

  for (unsigned int i = 0; i < sampleInputImg.get_height(); i++) {
    for (unsigned int j = 0; j < sampleInputImg.get_width(); j++) {
      char currentPixel = static_cast<char>(sampleInputImg[i][j]);
      char currentAlpha = 255;
      file.write(&currentPixel, sizeof(char));
      file.write(&currentPixel, sizeof(char));
      file.write(&currentPixel, sizeof(char));
      file.write(&currentAlpha, sizeof(char));
    }
  }

  file.close();

  sampleInputImg.write("images/mnistInput.png");*/

  return 0;
}
