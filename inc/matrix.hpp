#ifndef INCLUDED_MATRIX
#define INCLUDED_MATRIX

#include <ostream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <initializer_list>

class Matrix {
  private:
    unsigned int _height;
    unsigned int _width;
    std::vector<float> _data;

    std::uniform_real_distribution<float> _uniformDist{-1.0, 1.0};
  public:
    Matrix();
    Matrix(unsigned int height, unsigned int width);
    Matrix(const Matrix& m);
    Matrix(std::initializer_list< std::initializer_list<float> > initList);
    //~Matrix();
    void fill(float value);
    void randomFill(std::mt19937& rng);
    void randomFill(std::mt19937& rng, float low, float high);
    void randomBinomialFill(std::mt19937& rng, float p);
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

    bool equal(const Matrix& b) const;

    Matrix relu();
    Matrix reluInvDeriv();
    Matrix sigmoid();
    Matrix sigmoidInvDeriv();

    float absAvg();
};

std::ostream& operator<<(std::ostream& os, const Matrix& m);

#endif
