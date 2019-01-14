#ifndef INCLUDED_MATRIX
#define INCLUDED_MATRIX

#include <ostream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <omp.h>
#include "clops.hpp"

class Clops;

class Matrix {
  private:
    unsigned int _height;
    unsigned int _width;
    std::vector<float> _data;

    std::uniform_real_distribution<float> _uniformDist{-1.0, 1.0};

  public:
    static std::mt19937 rng;
    static Clops clops;

    Matrix();
    Matrix(unsigned int height, unsigned int width);
    Matrix(const Matrix& m);
    Matrix(std::initializer_list< std::initializer_list<float> > initList);
    //~Matrix();
    void fill(float value);
    void randomFill();
    void randomFill(float low, float high);
    void randomBinomialFill(float p);
    void set(unsigned int row, unsigned int col, float value);
    void set(unsigned int i, float value);
    float get(unsigned int row, unsigned int col) const;
    float get(unsigned int i) const;
    unsigned int getSize() const;
    float* getDataPointer();
    std::vector<float>& getDataVector();
    unsigned int getWidth() const;
    unsigned int getHeight() const;
    Matrix getSample(unsigned int height, unsigned int width) const;

    void identity();
    Matrix multiply(const Matrix& b) const;
    Matrix multiply(float b) const;
    Matrix elementMultiply(const Matrix& b) const;
    Matrix add(const Matrix& b) const;
    Matrix subtract(const Matrix& b) const;
    Matrix matVecAdd(const Matrix& v) const;
    Matrix transpose();
    Matrix sumAlongRows();

    Matrix gpuMultiply(Matrix& b);
    Matrix gpuMultiply(Matrix&& b);

    bool equal(const Matrix& b) const;

    Matrix relu();
    Matrix reluInvDeriv();
    Matrix sigmoid();
    Matrix sigmoidInvDeriv();
    Matrix softmax();
    Matrix softmaxInvDeriv();

    Matrix meanSquared(Matrix& y);
    Matrix meanSquaredDeriv(Matrix& y);
    Matrix crossEntropy(Matrix& y);
    Matrix crossEntropyDeriv(Matrix& y);

    float absAvg();
    void roundPrint();
};

std::ostream& operator<<(std::ostream& os, const Matrix& m);

#endif
