#include <cassert>
#include <iostream>
#include "../include/matrix.hpp"

void testAdd() {
  Matrix a(3,4);
  a.identity();
  Matrix b(4,3);
  b.fill(1.0);
  b.transpose();
  Matrix c{{2.0, 1.0, 1.0, 1.0}, {1.0, 2.0, 1.0, 1.0},  {1.0, 1.0, 2.0, 1.0}};
  //std::cout << a.add(b) << "=" << std::endl << c << std::endl;
  assert(a.add(b).equal(c));
}

void testMultiply() {
  Matrix a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  Matrix b{{6.0, 3.0}, {5.0, 2.0}, {4.0, 1.0}};
  Matrix c{{28.0, 10.0},{73.0, 28.0}};
  //std::cout << a.multiply(b) << "=" << std::endl << c << std::endl;
  assert(a.multiply(b).equal(c));
}

int main() {

  testAdd();
  testMultiply();

  return 0;
}
