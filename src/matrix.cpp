#include "matrix.hpp"

Matrix::Matrix() : _height(0), _width(0), _data(0, 0) {

}

Matrix::Matrix(unsigned int height, unsigned int width) : _height(height), _width(width), _data(width*height, 0) {

}

Matrix::Matrix(const Matrix& m) : _height(m.getHeight()), _width(m.getWidth()), _data(m.getHeight()*m.getWidth(), 0) {
  for (unsigned int i = 0; i < m.getSize(); i++) {
    _data[i] = m.get(i);
  }
}

Matrix::Matrix(std::initializer_list< std::initializer_list<float> > initList) : _height(0), _width(0), _data(0,0.0) {
  for (auto &row : initList) {
    for (auto &e : row) {
      _data.push_back(e);
      if (_height == 0) {
        _width++;
      }
    }
    _height++;
  }
}

//Matrix::~Matrix() {
//
//}

void Matrix::fill(float value) {
  for (auto &e : _data) {
    e = value;
  }
}

void Matrix::randomFill() {
  for (auto &e : _data) {
    e = _uniformDist(rng);
  }
}

void Matrix::randomFill(float low, float high) {
  for (auto &e : _data) {
    e = (_uniformDist(rng) + 1.0f)*0.5f*(high - low) + low;
  }
}

void Matrix::randomBinomialFill(float p) {
  for (auto &e : _data) {
    if (_uniformDist(rng) < p*2 - 1) {
      e = 0.0;
    } else {
      e = 1.0;
    }
  }
}

void Matrix::set(unsigned int row, unsigned int col, float value) {
  _data[row*_width + col] = value;
}

void Matrix::set(unsigned int i, float value) {
  _data[i] = value;
}

float Matrix::get(unsigned int row, unsigned int col) const {
  return _data[row*_width + col];
}

float Matrix::get(unsigned int i) const {
  return _data[i];
}

unsigned int Matrix::getSize() const {
  return _width*_height;
}

float* Matrix::getDataPointer() {
  return _data.data();
}

std::vector<float>& Matrix::getDataVector() {
  return _data;
}

unsigned int Matrix::getWidth() const {
  return _width;
}

unsigned int Matrix::getHeight() const {
  return _height;
}

Matrix Matrix::getSample(unsigned int height, unsigned int width) const {
  Matrix s(height, width);

  for (unsigned int i = 0; i < height; i++) {
    for (unsigned int j = 0; j < width; j++) {
      s.set(i, j, get(i*(_height / height), j*(_width / width)));
    }
  }

  return s;
}

void Matrix::identity() {
  fill(0.0);
  for (unsigned int i = 0; i < ((_width < _height) ? _width : _height); i++) {
    set(i, i, 1.0);
  }
}

Matrix Matrix::multiply(const Matrix& b) const {
  if (_width != b.getHeight()) {
    throw std::invalid_argument("invalid matrix dimensions for multiplication");
  }
  Matrix c(_height, b.getWidth());
  for (unsigned int i = 0; i < c.getHeight(); i++) {
    for (unsigned int j = 0; j < c.getWidth(); j++) {
      float v = 0;
      for (unsigned int k = 0; k < _width; k++) {
        v += get(i, k)*b.get(k, j);
      }
      c.set(i, j, v);
    }
  }
  return c;
}

Matrix Matrix::multiply(float b) const {
  Matrix ret(*this);

  for (auto &e : ret.getDataVector()) {
    e = e*b;
  }

  return ret;
}

Matrix  Matrix::elementMultiply(const Matrix& b) const {
  if (_width != b.getWidth() || _height !=  b.getHeight()) {
    throw std::invalid_argument("invalid matrix dimensions for elementMultiply");
  }

  Matrix c(_height, _width);

  for (unsigned int i = 0; i < getSize(); i++) {
    c.set(i, get(i) * b.get(i));
  }

  return c;
}

Matrix  Matrix::add(const Matrix& b) const {
  if (_width != b.getWidth() || _height !=  b.getHeight()) {
    throw std::invalid_argument("invalid matrix dimensions for addition");
  }

  Matrix c(_height, _width);

  for (unsigned int i = 0; i < getSize(); i++) {
    c.set(i, get(i) + b.get(i));
  }

  return c;
}

Matrix  Matrix::subtract(const Matrix& b) const {
  if (_width != b.getWidth() || _height !=  b.getHeight()) {
    throw std::invalid_argument("invalid matrix dimensions for subtraction");
  }

  Matrix c(_height, _width);

  for (unsigned int i = 0; i < getSize(); i++) {
    c.set(i, get(i) - b.get(i));
  }

  return c;
}


Matrix Matrix::matVecAdd(const Matrix& v) const {
  if (_height != v.getHeight() || v.getWidth() != 1) {
    throw std::invalid_argument("invalid matrix dimensions for matrix vector addition");
  }

  Matrix c(_height, _width);
  for (unsigned int i = 0; i < _height; i++) {
    for (unsigned int j = 0; j < _width; j++) {
      c.set(i, j, get(i, j) + v.get(i, 0));
    }
  }
  return c;
}

Matrix Matrix::transpose() {
  Matrix trans(_width, _height);
  for (unsigned int i = 0; i < _width; i++) {
    for (unsigned int j = 0; j < _height; j++) {
      trans.set(i,j, get(j, i));
    }
  }

  return trans;
  /*std::vector<float> newData(getSize(),0);
  for (unsigned int i = 0; i < _height; i++) {
    for (unsigned int j = 0; j < _width; j++) {
      unsigned int p = i*_width + j;
      unsigned int q = j*_height + i;
      newData[q] = _data[p];
    }
  }

  unsigned int tmp = _width;
  _width = _height;
  _height = tmp;

  for (unsigned int i = 0; i < getSize(); i++) {
    _data[i] = newData[i];
  }*/
}

bool Matrix::equal(const Matrix& b) const {
  if (_width != b.getWidth() || _height !=  b.getHeight()) {
    return false;
  }
  for (unsigned int i = 0; i < getSize(); i++) {
    if (get(i) != b.get(i)) {
      return false;
    }
  }
  return true;
}

Matrix Matrix::relu() {
  Matrix ret(*this);

  for (auto &e : ret.getDataVector()) {
    if (e < 0) {
      e = 0;
    }
  }
  return ret;
}

Matrix Matrix::reluInvDeriv() {
  Matrix ret(*this);

  for (auto &e : ret.getDataVector()) {
    e = e <= 0 ? 0 : 1;
  }

  return ret;
}

Matrix Matrix::sigmoid() {
  Matrix ret(*this);

  for (auto &e : ret.getDataVector()) {
    e = 1/(1 + static_cast<float>(exp(-e)));
  }

  return ret;
}

Matrix Matrix::sigmoidInvDeriv() {
  Matrix ret(*this);

  for (auto &e : ret.getDataVector()) {
    e = e*(1.0f - e);
  }

  return ret;
}

float Matrix::absAvg() {
  float sum = 0;

  for (auto &e : _data) {
    sum += abs(e);
  }

  return sum/static_cast<float>(_width*_height);
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
  for (unsigned int i = 0; i < m.getHeight(); i++) {
    for (unsigned int j = 0; j < m.getWidth(); j++) {
      os << m.get(i, j) << " ";
    }
    os << "\n";
  }
  return os;
}
