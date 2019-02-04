#ifndef INCLUDED_MNIST
#define INCLUDED_MNIST

#include <iostream>
#include "matrix.hpp"
#include <stdexcept>
#include <vector>
#include <string>
#include <fstream>
#include <Eigen/Dense>

class Mnist {
  private:
    std::string _name;
    int32_t _numLabels;
    int32_t _numImages;
    int32_t _imageHeight;
    int32_t _imageWidth;

    std::vector<char> _labelData;
    std::vector<char> _imageData;

    int32_t reverseInt(int32_t i);
  public:
    Mnist(std::string name);

    unsigned int getNumLabels();
    unsigned int getNumImages();
    unsigned int getImageHeight();
    unsigned int getImageWidth();

    std::vector<Eigen::MatrixXf> getLabelBatches(unsigned int batchSize);
    std::vector<Eigen::MatrixXf> getImageBatches(unsigned int batchSize);
    std::vector<Eigen::MatrixXf> getLabelBatches(unsigned int batchSize, unsigned int numBatches);
    std::vector<Eigen::MatrixXf> getImageBatches(unsigned int batchSize, unsigned int numBatches);

};

#endif
