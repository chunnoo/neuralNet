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
    Eigen::MatrixXf layerWeight = Eigen::MatrixXf::Random(layerSizes[i], layerSizes[i - 1]);

    Eigen::VectorXf layerBias = Eigen::VectorXf::Random(layerSizes[i]);

    _ws.push_back(layerWeight);
    _bs.push_back(layerBias);
  }
  _bs[_numLayers - 2].fill(0);
  //for (auto &e : _bs) {
  //  e.fill(0);
  //}
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
    Eigen::MatrixXf layerWeight = Eigen::MatrixXf::Random(layerSizesVec[i], layerSizesVec[i - 1]);

    Eigen::VectorXf layerBias = Eigen::VectorXf::Random(layerSizesVec[i]);

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
    Eigen::MatrixXf m(_layerSizes[i], _layerSizes[i-1]);
    for (unsigned int j = 0; j < m.rows(); j++) {
      for (unsigned int k = 0; k < m.cols(); k++) {
        float e;
        file.read(reinterpret_cast<char *>(&e), sizeof(float));
        m(j, k) = e;
      }
    }
    _ws.push_back(m);
  }

  for (unsigned int i = 1; i < _numLayers; i++) {
    Eigen::VectorXf v(_layerSizes[i]);
    for (unsigned int j = 0; j < v.size(); j++) {
      float e;
      file.read(reinterpret_cast<char *>(&e), sizeof(float));
      v(j) = e;
    }
    _bs.push_back(v);
  }

  file.close();
}

void NeuralNet::backPropagation(std::vector<Eigen::MatrixXf>& inputBatches, std::vector<Eigen::MatrixXf>& outputBatches, Loss loss, float alpha, float dropoutRate, unsigned int iterations, unsigned int iterModPrint) {
  if (inputBatches.size() != outputBatches.size()) {
    throw std::invalid_argument("inputBatches and outputBatches does not match");
  }

  for (unsigned int i = 0; i < iterations; i++) {
    std::vector<Eigen::MatrixXf> layers;
    Eigen::MatrixXf firstLayer = inputBatches[i%inputBatches.size()];
    layers.push_back(firstLayer);

    for (unsigned int j = 1; j < _numLayers; j++) {

      Eigen::MatrixXf layer = (_ws[j-1] * layers[j-1]).colwise() + _bs[j-1];

      if (_activations[j] == RELU) {
        layer = Operations::relu(layer);
      } else if (_activations[j] == SIGMOID) {
        layer = Operations::sigmoid(layer);
      } else if (_activations[j] == SOFTMAX) {
        layer = Operations::softmax(layer);
      }

      if (j < _numLayers - 1 && dropoutRate > 0) {
        Eigen::MatrixXf dropout = Eigen::MatrixXf::Random(layer.rows(), layer.cols());
        dropout = dropout.unaryExpr([=](float e){return static_cast<float>((e < dropoutRate*2 - 1 ? 0.0 : 1.0)*(1/(1 - dropoutRate)));});
        layer = layer.array() * dropout.array();
      }

      layers.push_back(layer);
    }

    //std::vector<Matrix> errors; //ordered form lastlayer error and backwards
    std::vector<Eigen::MatrixXf> deltas; //also ordered from last and backwards

    //Matrix lastError;
    Eigen::MatrixXf lastDelta;
    lastDelta = layers[_numLayers - 1] - (outputBatches[i%outputBatches.size()]);
    /*if (loss == MEANSQUARED) {
      //lastError = layers[_numLayers - 1].meanSquared(outputBatches[i%outputBatches.size()]);
      lastDelta = layers[_numLayers - 1].meanSquaredDeriv(outputBatches[i%outputBatches.size()]);
    } else if (loss == CROSSENTROPY) {
      //lastError = layers[_numLayers - 1].crossEntropy(outputBatches[i%outputBatches.size()]);
      lastDelta = layers[_numLayers - 1].crossEntropyDeriv(outputBatches[i%outputBatches.size()]);
    } else {
      //lastError = layers[_numLayers - 1].subtract(outputBatches[i%outputBatches.size()]);
      lastDelta = layers[_numLayers - 1].subtract(outputBatches[i%outputBatches.size()]);
    }

    if (_activations[_numLayers - 1] == RELU) {
      lastDelta = lastDelta.elementMultiply(layers[_numLayers - 1].reluInvDeriv());
    } else if (_activations[_numLayers - 1] == SIGMOID) {
      lastDelta = lastDelta.elementMultiply(layers[_numLayers - 1].sigmoidInvDeriv());
    } else if (_activations[_numLayers - 1] == SOFTMAX) {
      lastDelta = lastDelta.elementMultiply(layers[_numLayers - 1].softmaxInvDeriv());
    } else {
      lastDelta = lastDelta.elementMultiply(layers[_numLayers - 1]);
    }*/

    //errors.push_back(lastError);
    deltas.push_back(lastDelta);

    for (unsigned int j = _numLayers - 2; j > 0; j--) {
      Eigen::MatrixXf error = _ws[j].transpose() * deltas[_numLayers - 2 - j];
      //errors.push_back(error);

      Eigen::MatrixXf delta;
      if (_activations[j] == RELU) {
        //delta = errors[_numLayers - 1 - j].elementMultiply(layers[j].reluInvDeriv());
        delta = error.array() * Operations::reluInvDeriv(layers[j]).array();
      } else if (_activations[j] == SIGMOID) {
        delta = error.array() * Operations::sigmoidInvDeriv(layers[j]).array();
      } else if (_activations[j] == SOFTMAX) {
        delta = error.array() * Operations::reluInvDeriv(layers[j]).array();
        //This is not working
      } else {
        delta = error.array() * layers[j].array();
      }
      deltas.push_back(delta);
    }

    for (unsigned int j = 0; j < _numLayers - 1; j++) {
      _ws[j] = _ws[j] - ((deltas[_numLayers - 2 - j] * layers[j].transpose()) * alpha);
      _bs[j] = _bs[j] - (deltas[_numLayers - 2 - j].rowwise().sum() * alpha);
    }

    if (i % iterModPrint == 0) {
      std::cout << "Err: " << (layers[_numLayers - 1] - outputBatches[i%outputBatches.size()]).unaryExpr([](float e){return std::abs(e);}).mean() << std::endl;
    }
  }
}

Eigen::MatrixXf NeuralNet::use(Eigen::MatrixXf& input) {
  Eigen::MatrixXf output = input;
  for (unsigned int i = 0; i < _numLayers - 1; i++) {
    if (_activations[i + 1] == RELU) {
      output = Operations::relu((_ws[i] * output).colwise() + _bs[i]);
    } else if (_activations[i + 1] == SIGMOID) {
      output = Operations::sigmoid((_ws[i] * output).colwise() + _bs[i]);
    } else if (_activations[i + 1] == SOFTMAX) {
      output = Operations::softmax((_ws[i] * output).colwise() + _bs[i]);
    } else {
      output = ((_ws[i] * output).colwise() + _bs[i]);
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
    for (unsigned int i = 0; i < m.rows(); i++) {
      for (unsigned int j = 0; j < m.cols(); j++) {
        float e = m(i, j);
        file.write(reinterpret_cast<char *>(&e), sizeof(float));
      }
    }
  }

  for (auto &v : _bs) {
    for (unsigned int i = 0; i < v.size(); i++) {
      float e = v(i);
      file.write(reinterpret_cast<char *>(&e), sizeof(float));
    }
  }

  file.close();
}
