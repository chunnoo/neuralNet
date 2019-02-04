#ifndef INCLUDED_NEURALNET
#define INCLUDED_NEURALNET

#include <iostream>
#include <fstream>
#include "operations.hpp"
#include <stdexcept>
#include <vector>
#include <string>
#include <initializer_list>
#include <Eigen/Dense>
#include <cmath>

enum Activation {NONE, SIGMOID, RELU, SOFTMAX};
enum Loss {MEANSQUARED, CROSSENTROPY};

class NeuralNet {
  private:
    unsigned int _numLayers;
    std::vector<unsigned int> _layerSizes;
    std::vector<Eigen::MatrixXf> _ws;
    std::vector<Eigen::VectorXf> _bs;
    std::vector<Activation> _activations;

  public:
    NeuralNet(std::vector<unsigned int> layerSizes, std::vector<Activation> layerActivations);
    NeuralNet(std::initializer_list<unsigned int> layerSizes, std::initializer_list<Activation> layerActivations);
    NeuralNet(std::string filename);

    void backPropagation(std::vector<Eigen::MatrixXf>& inputBatches, std::vector<Eigen::MatrixXf>& outputBatches, Loss loss, float alpha, float dropoutRate, unsigned int iterations, unsigned int iterModPrint);

    Eigen::MatrixXf use(Eigen::MatrixXf& input);

    void printWeightsAndBiases();

    void save(std::string filename);

};

#endif
