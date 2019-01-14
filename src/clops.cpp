#include "clops.hpp"

#define MAX_SOURCE_SIZE (0x100000)

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

Clops::Clops(int maxSize) {
  _maxSize = maxSize;

  //load file
  FILE *fp;
  char *sourceStr;
  size_t sourceSize;

  fp = fopen("src/clops.cl", "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(0);
  }

  sourceStr = (char*)malloc(MAX_SOURCE_SIZE);
  sourceSize = fread(sourceStr, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
  //load file end

  cl_int ret;

  clGetPlatformIDs(1, &_platformId, &_numPlatforms);
  clGetDeviceIDs(_platformId, CL_DEVICE_TYPE_GPU, 1, &_deviceId, &_numDevices);

  _context = clCreateContext(nullptr, 1, &_deviceId, nullptr, nullptr, &ret);
  _commandQueue = clCreateCommandQueue(_context, _deviceId, 0, &ret);
  _program = clCreateProgramWithSource(_context, 1, (const char **)&sourceStr, (const size_t *)&sourceSize, &ret);

  char buildLog[MAX_SOURCE_SIZE];
  size_t *buildLogSize;
  clBuildProgram(_program, 1, &_deviceId, nullptr, nullptr, nullptr);
  clGetProgramBuildInfo(_program, _deviceId, CL_PROGRAM_BUILD_LOG, MAX_SOURCE_SIZE, buildLog, buildLogSize);

  std::cout << buildLog << std::endl;
  free(sourceStr);

  _numKernels = 2;
  _kernels = new cl_kernel[_numKernels];
  _numMemoryObjects = 3;
  _memoryObjects = new cl_mem[_numMemoryObjects];
  _constants = new int[2];

  _memoryObjects[0] = clCreateBuffer(_context, CL_MEM_READ_ONLY, _maxSize*sizeof(float), nullptr, &ret);
  _memoryObjects[1] = clCreateBuffer(_context, CL_MEM_READ_ONLY, _maxSize*sizeof(float), nullptr, &ret);
  _memoryObjects[2] = clCreateBuffer(_context, CL_MEM_WRITE_ONLY, _maxSize*sizeof(float), nullptr, &ret);

  _kernels[0] = clCreateKernel(_program, "add", &ret);
  _kernels[1] = clCreateKernel(_program, "multiply", &ret);

  clSetKernelArg(_kernels[0], 0, sizeof(cl_mem), (void *)&_memoryObjects[0]);
  clSetKernelArg(_kernels[0], 1, sizeof(cl_mem), (void *)&_memoryObjects[1]);
  clSetKernelArg(_kernels[0], 2, sizeof(cl_mem), (void *)&_memoryObjects[2]);

  clSetKernelArg(_kernels[1], 0, sizeof(cl_mem), (void *)&_memoryObjects[0]);
  clSetKernelArg(_kernels[1], 1, sizeof(cl_mem), (void *)&_memoryObjects[1]);
  clSetKernelArg(_kernels[1], 2, sizeof(cl_mem), (void *)&_memoryObjects[2]);
  clSetKernelArg(_kernels[1], 3, sizeof(int), _constants);
  clSetKernelArg(_kernels[1], 4, sizeof(int), (_constants + 1));

}

Clops::~Clops() {
  clFlush(_commandQueue);
  clFinish(_commandQueue);

  for (int i = 0; i < _numKernels; i++) {
    clReleaseKernel(_kernels[i]);
  }
  delete[] _kernels;

  clReleaseProgram(_program);

  for (int i = 0; i < _numMemoryObjects; i++) {
    clReleaseMemObject(_memoryObjects[i]);
  }
  delete[] _memoryObjects;

  clReleaseCommandQueue(_commandQueue);
  clReleaseContext(_context);

  delete[] _constants;
}

void Clops::add(Matrix& a, Matrix& b, Matrix& c) {
  if (a.getHeight() != b.getHeight() || a.getHeight() != c.getHeight() || a.getWidth() != b.getWidth() || a.getWidth() != c.getWidth()) {
    throw std::invalid_argument("invalid matrix dimensions for addition");
  }

  int workSize = a.getWidth()*a.getHeight();
  size_t globalWorkOffset = 0;
  size_t globalWorkSize = workSize;

  clEnqueueWriteBuffer(_commandQueue, _memoryObjects[0], CL_TRUE, 0, workSize*sizeof(float), a.getDataPointer(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(_commandQueue, _memoryObjects[1], CL_TRUE, 0, workSize*sizeof(float), b.getDataPointer(), 0, nullptr, nullptr);

  /*clSetKernelArg(_kernels[0], 0, sizeof(cl_mem), (void *)&_memoryObjects[0]);
  clSetKernelArg(_kernels[0], 1, sizeof(cl_mem), (void *)&_memoryObjects[1]);
  clSetKernelArg(_kernels[0], 2, sizeof(cl_mem), (void *)&_memoryObjects[2]);*/

  clEnqueueNDRangeKernel(_commandQueue, _kernels[0], 1, &globalWorkOffset, &globalWorkSize, nullptr, 0, nullptr, nullptr);

  clEnqueueReadBuffer(_commandQueue, _memoryObjects[2], CL_TRUE, 0, workSize*sizeof(float), c.getDataPointer(), 0, nullptr, nullptr);

  clFlush(_commandQueue);
  clFinish(_commandQueue);
}

void Clops::multiply(Matrix& a, Matrix& b, Matrix& c) {
  if (a.getWidth() != b.getHeight() || a.getHeight() != c.getHeight() || b.getWidth() != c.getWidth()) {
    throw std::invalid_argument("invalid matrix dimensions for multiplication");
  }

  size_t globalWorkSize[2];
  globalWorkSize[0] = a.getHeight();
  globalWorkSize[1] = b.getWidth();

  _constants[0] = c.getWidth();
  _constants[1] = a.getWidth();
  /*clSetKernelArg(_kernels[1], 0, sizeof(cl_mem), (void *)&_memoryObjects[0]);
  clSetKernelArg(_kernels[1], 1, sizeof(cl_mem), (void *)&_memoryObjects[1]);
  clSetKernelArg(_kernels[1], 2, sizeof(cl_mem), (void *)&_memoryObjects[2]);*/
  clSetKernelArg(_kernels[1], 3, sizeof(int), _constants);
  clSetKernelArg(_kernels[1], 4, sizeof(int), (_constants + 1));

  clEnqueueWriteBuffer(_commandQueue, _memoryObjects[0], CL_TRUE, 0, a.getWidth()*a.getHeight()*sizeof(float), a.getDataPointer(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(_commandQueue, _memoryObjects[1], CL_TRUE, 0, b.getWidth()*b.getHeight()*sizeof(float), b.getDataPointer(), 0, nullptr, nullptr);

  clEnqueueNDRangeKernel(_commandQueue, _kernels[1], 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);

  clEnqueueReadBuffer(_commandQueue, _memoryObjects[2], CL_TRUE, 0, c.getWidth()*c.getHeight()*sizeof(float), c.getDataPointer(), 0, nullptr, nullptr);

  clFlush(_commandQueue);
  clFinish(_commandQueue);
}
