#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <vector>

// Permute only helps when exchanging data in one wave.......

const static int width = 8;
const static int height = 8;
const static int tile_dim = 8;

__global__ void transpose_lds_kernel(float *in, float *out, int width,
                                     int height) {

  int in_index = threadIdx.y * width + threadIdx.x;
  int out_index = threadIdx.x * width + threadIdx.y;

  float entry = in[in_index];
  float others = __hip_ds_bpermutef(out_index*4, entry);

  out[in_index] = others;
}


int main() {
  std::vector<float> matrix_in;
  std::vector<float> matrix_out;

  matrix_in.resize(width * height);
  matrix_out.resize(width * height);

  for (int i = 0; i < width * height; i++) {
    matrix_in[i] = (float)rand() / (float)RAND_MAX;
  }

  // for(size_t i = 0; i < width * height; i++)
  // {
  //     std::cout << "[" << i << "]" << matrix_in[i] << "\t";
  //     if((i + 1) % width == 0)
  //     {
  //         std::cout << std::endl;
  //     }
  // }

  float *d_in;
  float *d_out;

  hipMalloc((void **)&d_in, width * height * sizeof(float));
  hipMalloc((void **)&d_out, width * height * sizeof(float));

  hipMemcpy(d_in, matrix_in.data(), width * height * sizeof(float),
            hipMemcpyHostToDevice);

  printf("Setup complete. Launching kernel \n");
  int block_x = width / tile_dim;
  int block_y = height / tile_dim;

  hipLaunchKernelGGL(transpose_lds_kernel, dim3(block_x, block_y),
                      dim3(tile_dim, tile_dim), 0, 0, d_in, d_out, width,
                      height);

   hipDeviceSynchronize();

   printf("Kernel execution complete \n");

   hipMemcpy(matrix_out.data(), d_out, width * height * sizeof(float),
            hipMemcpyDeviceToHost);


    // for(size_t i = 0; i < width * height; i++)
    // {
    //     std::cout << "[" << i << "]" << matrix_out[i] << "\t";
    //     if((i + 1) % width == 0)
    //         std::cout << std::endl;
    // }


  return 0;
}