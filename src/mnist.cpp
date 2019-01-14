#include "mnist.hpp"

int32_t Mnist::reverseInt(int32_t i) {
    unsigned char c1, c2, c3, c4;

    c1 = static_cast<unsigned char>(i & 255);
    c2 = static_cast<unsigned char>((i >> 8) & 255);
    c3 = static_cast<unsigned char>((i >> 16) & 255);
    c4 = static_cast<unsigned char>((i >> 24) & 255);

    return (static_cast<int32_t>(c1) << 24) + (static_cast<int32_t>(c2) << 16) + (static_cast<int32_t>(c3) << 8) + c4;
}

Mnist::Mnist(std::string name) : _name(name) {
  std::string labelFileName = "mnist/" + _name + "-labels.idx1-ubyte";

  std::ifstream labelFile(labelFileName, std::ios::binary);

  int32_t labelMagNum;

  labelFile.seekg(0, std::ios::beg);
  labelFile.read(reinterpret_cast<char *>(&labelMagNum), sizeof(int32_t));
  labelMagNum = reverseInt(labelMagNum);
  labelFile.read(reinterpret_cast<char *>(&_numLabels), sizeof(int32_t));
  _numLabels = reverseInt(_numLabels);

  _labelData.reserve(static_cast<unsigned int>(_numLabels));

  labelFile.read(_labelData.data(), static_cast<unsigned int>(_numLabels)*sizeof(char));

  std::string imageFileName = "mnist/" + _name + "-images.idx3-ubyte";

  std::ifstream imageFile(imageFileName, std::ios::binary | std::ios::ate);

  int32_t imageMagNum;

  imageFile.seekg(0, std::ios::beg);
  imageFile.read(reinterpret_cast<char *>(&imageMagNum), sizeof(int32_t));
  imageMagNum = reverseInt(imageMagNum);
  imageFile.read(reinterpret_cast<char *>(&_numImages), sizeof(int32_t));
  _numImages = reverseInt(_numImages);
  imageFile.read(reinterpret_cast<char *>(&_imageHeight), sizeof(int32_t));
  _imageHeight = reverseInt(_imageHeight);
  imageFile.read(reinterpret_cast<char *>(&_imageWidth), sizeof(int32_t));
  _imageWidth = reverseInt(_imageWidth);

  _imageData.reserve(static_cast<unsigned int>(_numImages*_imageHeight*_imageWidth));

  imageFile.read(_imageData.data(), static_cast<unsigned int>(_numImages*_imageHeight*_imageWidth)*sizeof(char));

  labelFile.close();
  imageFile.close();
}

unsigned int Mnist::getNumLabels() {
  return static_cast<unsigned int>(_numLabels);
}

unsigned int Mnist::getNumImages() {
  return static_cast<unsigned int>(_numImages);
}

unsigned int Mnist::getImageHeight() {
  return static_cast<unsigned int>(_imageHeight);
}

unsigned int Mnist::getImageWidth() {
  return static_cast<unsigned int>(_imageWidth);
}

std::vector<Matrix> Mnist::getLabelBatches(unsigned int batchSize) {
  std::vector<Matrix> matrices;

  for (unsigned int i = 0; i < static_cast<unsigned int>(_numLabels) / batchSize; i++) {
    Matrix labelBatch(10, batchSize);
    labelBatch.fill(0);

    for (unsigned int j = 0; j < batchSize; j++) {
      labelBatch.set(static_cast<unsigned int>(_labelData[batchSize*i + j]), j, 1);
    }

    matrices.push_back(labelBatch);
  }

  return matrices;
}

std::vector<Matrix> Mnist::getLabelBatches(unsigned int batchSize, unsigned int numBatches) {
  std::vector<Matrix> matrices;

  for (unsigned int i = 0; i < numBatches; i++) {
    Matrix labelBatch(10, batchSize);
    labelBatch.fill(0);

    for (unsigned int j = 0; j < batchSize; j++) {
      labelBatch.set(static_cast<unsigned int>(_labelData[batchSize*i + j]), j, 1);
    }

    matrices.push_back(labelBatch);
  }

  return matrices;
}

std::vector<Matrix> Mnist::getImageBatches(unsigned int batchSize) {
  std::vector<Matrix> matrices;

  for (unsigned int i = 0; i < static_cast<unsigned int>(_numImages) / batchSize; i++) {
    Matrix imageBatch(static_cast<unsigned int>(_imageHeight*_imageWidth), batchSize);

    for (unsigned int j = 0; j < batchSize; j++) {
      for (unsigned int k = 0; k < static_cast<unsigned int>(_imageHeight*_imageWidth); k++) {
        imageBatch.set(k, j, static_cast<float>(_imageData[batchSize*static_cast<unsigned int>(_imageHeight*_imageWidth)*i + static_cast<unsigned int>(_imageHeight*_imageWidth)*j + k])/255);
      }
    }

    matrices.push_back(imageBatch);
  }

  return matrices;
}

std::vector<Matrix> Mnist::getImageBatches(unsigned int batchSize, unsigned int numBatches) {
  std::vector<Matrix> matrices;

  for (unsigned int i = 0; i < numBatches; i++) {
    Matrix imageBatch(static_cast<unsigned int>(_imageHeight*_imageWidth), batchSize);

    for (unsigned int j = 0; j < batchSize; j++) {
      for (unsigned int k = 0; k < static_cast<unsigned int>(_imageHeight*_imageWidth); k++) {
        imageBatch.set(k, j, static_cast<float>(_imageData[batchSize*static_cast<unsigned int>(_imageHeight*_imageWidth)*i + static_cast<unsigned int>(_imageHeight*_imageWidth)*j + k])/255);
      }
    }

    matrices.push_back(imageBatch);
  }

  return matrices;
}
