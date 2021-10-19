#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <vector>

// Permute only helps when exchanging data in one wave.......
const static int width = 8;
const static int height = 8;
const static int tile_dim = 8;
const static int nbatches = 100000;

__global__ void transpose_perm_kernel(float *in, float *out, int width,
                                     int height, int batch) {

  int in_index = threadIdx.y * width + threadIdx.x;
  int trans_index = threadIdx.x * width + threadIdx.y;

  const int matrixStrides = width * height;
  while(in_index < matrixStrides * batch)
  {
    float entry = in[in_index];
    float others = __hip_ds_bpermutef(trans_index*4, entry);
    out[in_index] = others;
    in_index += matrixStrides;
  }
}

__global__ void transpose_lds_kernel(float *in, float *out, int width,
                                     int height, int batch) {
  __shared__ float tile[tile_dim][tile_dim];

  int x_tile_index = blockIdx.x * tile_dim;
  int y_tile_index = blockIdx.y * tile_dim;

  int in_index = (y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
  int out_index = (x_tile_index + threadIdx.y) * height + (y_tile_index + threadIdx.x);

  const int matrixStrides = width * height;
  while(in_index < matrixStrides * batch)
  {
    tile[threadIdx.y][threadIdx.x] = in[in_index];
    __syncthreads();

    out[out_index] = tile[threadIdx.x][threadIdx.y];
    __syncthreads();

    in_index += matrixStrides;
    out_index += matrixStrides;
  }
}


int main() {
  std::vector<float> matrix_in;
  std::vector<float> matrix_out;

  matrix_in.resize(width * height * nbatches);
  matrix_out.resize(width * height * nbatches);

  for(int b = 0; b < nbatches; ++b)
  {
    for (int i = 0; i < width * height; i++) {
      matrix_in[(width * height) * b + i] = (float)rand() / (float)RAND_MAX;
      // matrix_in[(width * height) * b + i] = i;
    }
  }

#if VERBOSE
  for(int b = 0; b < nbatches; ++b)
  {
    std::cout << "========= Matrix : " << b << " ============" << std::endl;
    for(size_t i = 0; i < width * height; i++)
    {
        std::cout << "[" << i << "]" << matrix_in[(width * height) * b + i] << "\t";
        if((i + 1) % width == 0)
            std::cout << std::endl;
    }
  }
#endif

  float *d_in;
  float *d_out;

  hipMalloc((void **)&d_in, width * height * nbatches * sizeof(float));
  hipMalloc((void **)&d_out, width * height * nbatches * sizeof(float));

  hipMemcpy(d_in, matrix_in.data(), width * height * nbatches * sizeof(float),
            hipMemcpyHostToDevice);

  printf("Setup complete. Launching kernel \n");
  int block_x = width / tile_dim;
  int block_y = height / tile_dim;

#ifdef USE_PERMUTE
  hipLaunchKernelGGL(transpose_perm_kernel, dim3(block_x, block_y),
                      dim3(tile_dim, tile_dim), 0, 0, d_in, d_out, width,
                      height, nbatches);
#else
  hipLaunchKernelGGL(transpose_lds_kernel, dim3(block_x, block_y),
                      dim3(tile_dim, tile_dim), 0, 0, d_in, d_out, width,
                      height, nbatches);
#endif

   hipDeviceSynchronize();

   printf("Kernel execution complete \n");

   hipMemcpy(matrix_out.data(), d_out, width * height * nbatches * sizeof(float),
            hipMemcpyDeviceToHost);

#if VERBOSE
  for(int b = 0; b < nbatches; ++b)
  {
    std::cout << "========= Transposed : " << b << " ============" << std::endl;
    for(size_t i = 0; i < width * height; i++)
    {
        std::cout << "[" << i << "]" << matrix_out[(width * height) * b + i] << "\t";
        if((i + 1) % width == 0)
            std::cout << std::endl;
    }
  }
#endif

  return 0;
}