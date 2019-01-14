#ifndef CLOPS_INCLUDED
#define CLOPS_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <stdexcept>
#include "matrix.hpp"
#include <OpenCL/opencl.h>

class Matrix;

class Clops {
  private:
    int _maxSize;
    cl_platform_id _platformId;
    cl_device_id _deviceId;
    cl_uint _numDevices;
    cl_uint _numPlatforms;
    cl_context _context;
    cl_command_queue _commandQueue;
    cl_program _program;
    cl_kernel* _kernels;
    cl_int _numKernels;
    cl_mem* _memoryObjects;
    cl_int _numMemoryObjects;
    int* _constants;
  public:
    Clops(int maxSize);
    ~Clops();

    void add(Matrix& a, Matrix& b, Matrix& c);
    void multiply(Matrix& a, Matrix& b, Matrix& c);
};

#endif
