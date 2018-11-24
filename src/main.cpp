#include <iostream>
#include <random>
#include <ctime>
#include <algorithm>
#include <png++/png.hpp>
#include "matrix.hpp"

int main() {

  std::mt19937 rng(static_cast<unsigned int>(time(0)));

  const unsigned int batchSize = 16;
  const unsigned int trainingBatches = 16;
  const float alpha = 0.001f;
  const float dropoutRate = 0.0f;

  std::vector<Matrix> xs;
  std::vector<Matrix> ys;

  for (unsigned int i = 0; i < trainingBatches; i++) {
    Matrix x(2, batchSize);
    Matrix y(2, batchSize);
    x.randomFill(rng, -10.0, 10.0);
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

  Matrix w0(4,2);
  w0.randomFill(rng);
  Matrix w1(4,4);
  w1.randomFill(rng);
  Matrix w2(2,4);
  w2.randomFill(rng);
  Matrix b1(w0.getHeight(), 1);
  b1.randomFill(rng, 0.0, 1.0);
  Matrix b2(w1.getHeight(), 1);
  b2.randomFill(rng, 0.0, 1.0);
  Matrix b3(w2.getHeight(), 1);
  b3.randomFill(rng, 0.0, 1.0);


  for (unsigned int i = 0; i < 1024*64; i++) {
    Matrix a0 = xs[i%trainingBatches];

    Matrix a1 = w0.multiply(a0).matVecAdd(b1).relu();
    Matrix dropout1(a1.getHeight(), a1.getWidth());
    dropout1.randomBinomialFill(rng, dropoutRate);
    dropout1 = dropout1.multiply(1/(1-dropoutRate));
    a1 = a1.elementMultiply(dropout1);

    Matrix a2 = w1.multiply(a1).matVecAdd(b2).relu();
    /*Matrix dropout2(a2.getHeight(), a2.getWidth());
    dropout2.randomBinomialFill(rng, dropoutRate);
    dropout2 = dropout2.multiply(1/(1-dropoutRate));
    a2 = a2.elementMultiply(dropout2);*/

    Matrix a3 = w2.multiply(a2).matVecAdd(b3).sigmoid();

    Matrix e3 = a3.subtract(ys[i%trainingBatches]);
    Matrix delta3 = e3.elementMultiply(a3.sigmoidInvDeriv());

    Matrix e2 = w2.transpose().multiply(delta3);
    Matrix delta2 = e2.elementMultiply(a2.reluInvDeriv());

    Matrix e1 = w1.transpose().multiply(delta2);
    Matrix delta1 = e1.elementMultiply(a1.reluInvDeriv());

    w0 = w0.subtract(delta1.multiply(a0.transpose()).multiply(alpha));
    w1 = w1.subtract(delta2.multiply(a1.transpose()).multiply(alpha));
    w2 = w2.subtract(delta3.multiply(a2.transpose()).multiply(alpha));

    /*std::cout << a0 << std::endl;
    std::cout << a1 << std::endl;
    std::cout << a2 << std::endl;
    std::cout << a3 << std::endl;
    std::cout << e3 << std::endl;
    std::cout << delta3 << std::endl;
    std::cout << e2 << std::endl;
    std::cout << delta2 << std::endl;
    std::cout << e1 << std::endl;
    std::cout << delta1 << std::endl;
    std::cout << w0 << std::endl;
    std::cout << w1 << std::endl;
    std::cout << w2 << std::endl;*/
    if (i % 1024 == 0) {
      std::cout << "Err: " << e3.absAvg() << std::endl;
    }
  }

  unsigned int imgWidth = 128;
  unsigned int imgHeight = 128;

  unsigned int testSize = 1024;
  Matrix testX(2, testSize);
  testX.randomFill(rng, -10.0, 10.0);
  Matrix testY = w2.multiply(w1.multiply(w0.multiply(testX).matVecAdd(b1).relu()).matVecAdd(b2).relu()).matVecAdd(b3).sigmoid();

  //std::cout << x << std::endl << y << std::endl;

  png::image<png::rgb_pixel> imgX(imgWidth,imgHeight);

  for (unsigned int i = 0; i < trainingBatches; i++) {
    for (unsigned int j = 0; j < batchSize; j++) {
      imgX[static_cast<unsigned int>((xs[i].get(0, j)*0.1f + 1)*static_cast<float>(imgWidth)*0.5f)][static_cast<unsigned int>((xs[i].get(1, j)*0.1f + 1)*static_cast<float>(imgHeight)*0.5f)] = png::rgb_pixel(static_cast<unsigned char>(ys[i].get(0,j)*255), static_cast<unsigned char>(ys[i].get(1,j)*255), 0);
    }
  }

  imgX.write("images/imgX.png");

  png::image<png::rgb_pixel> imgY(imgWidth,imgHeight);

  for (unsigned int i = 0; i < testSize; i++) {
    imgY[static_cast<unsigned int>((testX.get(0, i)*0.1f + 1)*static_cast<float>(imgWidth)*0.5f)][static_cast<unsigned int>((testX.get(1, i)*0.1f + 1)*static_cast<float>(imgHeight)*0.5f)] = png::rgb_pixel(static_cast<unsigned char>(testY.get(0,i)*255), static_cast<unsigned char>(testY.get(1,i)*255), 0);
  }

  imgY.write("images/imgY.png");

  return 0;
}
