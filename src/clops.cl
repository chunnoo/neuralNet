__kernel void add(__global const float *a, __global const float *b, __global float *c) {

  int i = get_global_id(0);

  c[i] = a[i] + b[i];

}

__kernel void multiply(__global const float *a, __global const float *b, __global float *c, const int width, const int depth) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  //c[i*width + j] = i*width + j;

  float v = 0;
  for (int k = 0; k < depth; k++) {
    v += a[i*depth + k]*b[k*width + j];
  }

  c[i*width + j] = v;
}
