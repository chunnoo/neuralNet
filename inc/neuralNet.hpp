#ifndef INCLUDED_NEURALNET
#define INCLUDED_NEURALNET

#include <iostream>
#include <matrix.hpp>
#include <stdexcept>
#include <vector>
#include <initializer_list>

enum Activation {NONE, SIGMOID, RELU};

class NeuralNet {
  private:
    unsigned int _numLayers;
    std::vector<Matrix> _ws;
    std::vector<Matrix> _bs;
    std::vector<Activation> _activations;

  public:
    NeuralNet(std::vector<unsigned int> layerSizes, std::vector<Activation> layerActivations);
    NeuralNet(std::initializer_list<unsigned int> layerSizes, std::initializer_list<Activation> layerActivations);

    void backPropagation(std::vector<Matrix>& inputBatches, std::vector<Matrix>& outputBatches, float alpha, float dropoutRate, unsigned int iterations, unsigned int iterModPrint);

    Matrix use(Matrix& input);

};

#endif