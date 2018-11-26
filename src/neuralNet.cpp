#include "neuralNet.hpp"

NeuralNet::NeuralNet(std::vector<unsigned int> layerSizes, std::vector<Activation> layerActivations) : _numLayers(static_cast<unsigned int>(layerSizes.size())) {
  if (layerSizes.size() == layerActivations.size() + 1) {
    _activations.push_back(NONE);
  } else if (layerSizes.size() != layerActivations.size()) {
    throw std::invalid_argument("layerSizes and layerActivations does not match");
  }

  for (auto &e : layerSizes) {
    _layerSizes.push_back(e);
  }

  for (auto &e : layerActivations) {
    _activations.push_back(e);
  }

  for (unsigned int i = 1; i < _numLayers; i++) {
    Matrix layerWeight(layerSizes[i], layerSizes[i - 1]);
    layerWeight.randomFill();

    Matrix layerBias(layerSizes[i], 1);
    layerBias.randomFill();

    _ws.push_back(layerWeight);
    _bs.push_back(layerBias);
  }
  _bs[_numLayers - 2].fill(0);
}

NeuralNet::NeuralNet(std::initializer_list<unsigned int> layerSizes, std::initializer_list<Activation> layerActivations) : _numLayers(static_cast<unsigned int>(layerSizes.size())) {
  if (layerSizes.size() == layerActivations.size() + 1) {
    _activations.push_back(NONE);
  } else if (layerSizes.size() != layerActivations.size()) {
    throw std::invalid_argument("layerSizes and layerActivations does not match");
  }

  for (auto &e : layerSizes) {
    _layerSizes.push_back(e);
  }

  for (auto &e : layerActivations) {
    _activations.push_back(e);
  }

  std::vector<unsigned int> layerSizesVec;
  for (auto &e : layerSizes) {
    layerSizesVec.push_back(e);
  }

  for (unsigned int i = 1; i < _numLayers; i++) {
    Matrix layerWeight(layerSizesVec[i], layerSizesVec[i - 1]);
    layerWeight.randomFill();

    Matrix layerBias(layerSizesVec[i], 1);
    layerBias.randomFill();

    _ws.push_back(layerWeight);
    _bs.push_back(layerBias);
  }

  _bs[_numLayers - 2].fill(0);
}

NeuralNet::NeuralNet(std::string filename) {
  std::string fullFileName = "networks/" + filename;

  std::ifstream file(fullFileName, std::ios::binary);

  file.seekg(0, std::ios::beg);

  file.read(reinterpret_cast<char *>(&_numLayers), sizeof(unsigned int));

  for (unsigned int i = 0; i < _numLayers; i++) {
    unsigned int e;
    file.read(reinterpret_cast<char *>(&e), sizeof(unsigned int));
    _layerSizes.push_back(e);
  }

  for (unsigned int i = 0; i < _numLayers; i++) {
    Activation e;
    file.read(reinterpret_cast<char *>(&e), sizeof(Activation));
    _activations.push_back(e);
  }

  for (unsigned int i = 1; i < _numLayers; i++) {
    Matrix m(_layerSizes[i], _layerSizes[i-1]);
    for (auto &e : m.getDataVector()) {
      file.read(reinterpret_cast<char *>(&e), sizeof(float));
    }
    _ws.push_back(m);
  }

  for (unsigned int i = 1; i < _numLayers; i++) {
    Matrix m(_layerSizes[i], 1);
    for (auto &e : m.getDataVector()) {
      file.read(reinterpret_cast<char *>(&e), sizeof(float));
    }
    _bs.push_back(m);
  }

  file.close();
}

void NeuralNet::backPropagation(std::vector<Matrix>& inputBatches, std::vector<Matrix>& outputBatches, float alpha, float dropoutRate, unsigned int iterations, unsigned int iterModPrint) {
  if (inputBatches.size() != outputBatches.size()) {
    throw std::invalid_argument("inputBatches and outputBatches does not match");
  }

  for (unsigned int i = 0; i < iterations; i++) {
    std::vector<Matrix> layers;
    Matrix firstLayer = inputBatches[i%inputBatches.size()];
    layers.push_back(firstLayer);

    for (unsigned int j = 1; j < _numLayers; j++) {

      Matrix layer = _ws[j-1].multiply(layers[j-1]).matVecAdd(_bs[j-1]);

      if (_activations[j] == RELU) {
        layer = layer.relu();
      } else if (_activations[j] == SIGMOID) {
        layer = layer.sigmoid();
      }

      if (j != _numLayers - 1) {
        Matrix dropout(layer.getHeight(), layer.getWidth());
        dropout.randomBinomialFill(dropoutRate);
        dropout = dropout.multiply(1/(1 - dropoutRate));
        layer = layer.elementMultiply(dropout);
      }

      layers.push_back(layer);
    }

    std::vector<Matrix> errors; //ordered form lastlayer error and backwards
    std::vector<Matrix> deltas; //also ordered from last and backwards
    Matrix lastError = layers[_numLayers - 1].subtract(outputBatches[i%outputBatches.size()]);
    errors.push_back(lastError);

    for (unsigned int j = _numLayers - 1; j > 0; j--) {
      if (j != _numLayers - 1) {
        Matrix error = _ws[j].transpose().multiply(deltas[_numLayers - 2 - j]);
        errors.push_back(error);
      }

      Matrix delta;
      if (_activations[j] == RELU) {
        delta = errors[_numLayers - 1 - j].elementMultiply(layers[j].reluInvDeriv());
      } else if (_activations[j] == SIGMOID) {
        delta = errors[_numLayers - 1 - j].elementMultiply(layers[j].sigmoidInvDeriv());
      } else {
        delta = errors[_numLayers - 1 - j].elementMultiply(layers[j]);
      }
      deltas.push_back(delta);
    }

    for (unsigned int j = 0; j < _numLayers - 1; j++) {
      _ws[j] = _ws[j].subtract(deltas[_numLayers - 2 - j].multiply(layers[j].transpose()).multiply(alpha));
    }

    if (i % iterModPrint == 0) {
      std::cout << "Err: " << errors[0].absAvg() << std::endl;
    }
  }
}

Matrix NeuralNet::use(Matrix& input) {
  Matrix output = input;
  for (unsigned int i = 0; i < _numLayers - 1; i++) {
    if (_activations[i + 1] == RELU) {
      output = _ws[i].multiply(output).matVecAdd(_bs[i]).relu();
    } else if (_activations[i + 1] == SIGMOID) {
      output = _ws[i].multiply(output).matVecAdd(_bs[i]).sigmoid();
    } else {
      output = _ws[i].multiply(output).matVecAdd(_bs[i]);
    }
  }
  return output;
}

void NeuralNet::printWeightsAndBiases() {
  std::cout << "Weights" << std::endl;
  for (auto &e : _ws) {
    std::cout << e << std::endl;
  }

  std::cout << "Biases" << std::endl;
  for (auto &e : _bs) {
    std::cout << e << std::endl;
  }
}

void NeuralNet::save(std::string filename) {
  std::string fullFileName = "networks/" + filename;

  std::ofstream file(fullFileName, std::ios::binary);

  //file.seekg(0, std::ios::beg);

  file.write(reinterpret_cast<char *>(&_numLayers), sizeof(unsigned int));

  for (auto &e : _layerSizes) {
    file.write(reinterpret_cast<char *>(&e), sizeof(unsigned int));
  }

  for (auto &e : _activations) {
    file.write(reinterpret_cast<char *>(&e), sizeof(Activation));
  }

  for (auto &m : _ws) {
    for (auto &e : m.getDataVector()) {
      file.write(reinterpret_cast<char *>(&e), sizeof(float));
    }
  }

  for (auto &m : _bs) {
    for (auto &e : m.getDataVector()) {
      file.write(reinterpret_cast<char *>(&e), sizeof(float));
    }
  }

  file.close();
}
