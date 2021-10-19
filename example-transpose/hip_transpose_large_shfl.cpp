#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <vector>

const static int width = 32;
const static int height = 32;
const static int tile_dim = 32;
const static int batch = 10000; // how many batches

// row major......
__global__ void transpose_shift_kernel(float *in, float *out, int width,
                                     int height) {

  float r[tile_dim];  // each thread reads a column = tile dim elements

  int x_tile_index = blockIdx.x * tile_dim;
  int y_tile_index = blockIdx.y * tile_dim;
  int col_index = (y_tile_index * width) + (x_tile_index + threadIdx.x); // the start index of this col in this batch

  const int tid = threadIdx.x;

    r[0] = in[col_index];

    // <!-- START: Stage One (Horizontal Rotations) -->
    int addr = threadIdx.x % tile_dim;
    const int lane = (tid + 1) % tile_dim;
    #pragma unroll
    for(int elemID = 1; elemID < tile_dim; ++elemID)
    {
      r[elemID] = in[col_index + width * elemID];
      addr      = __shfl(addr, lane, 32);
      r[elemID] = __shfl(r[elemID], addr, 32);
    }
    // <!-- END: Stage One (Horizontal Rotations) -->

    // <!-- START: Stage Two (Vertical Rotations) -->
    float t[tile_dim];
    for (int k = 0; k < tile_dim; ++k)
      t[k] = r[(tile_dim - tid + k) % tile_dim];
    // <!-- END: Stage Two (Vertical Rotations) -->

    // <!-- START: Stage Three (Horizontal Rotations) -->
    addr = (tile_dim - tid) % tile_dim;
    const int lane2 = (tid + tile_dim - 1) % tile_dim;
    #pragma unroll
    for(int elemID = 0; elemID < tile_dim; ++elemID)
    {
      t[elemID] = __shfl(t[elemID], addr, 32);
      addr      = __shfl(addr, lane2, 32);
      out[col_index + width * elemID] = t[elemID];
    }
    // <!-- END: Stage Three (Horizontal Rotations) -->

#if 0
    r[0] = in[col_index];
    r[1] = in[col_index + width * 1];
    r[2] = in[col_index + width * 2];
    r[3] = in[col_index + width * 3];

    // <!-- START: Stage One (Horizontal Rotations) -->
    int addr = threadIdx.x % 4;
    addr = __shfl(addr, (tid + 1) % 4);
    r[1] = __shfl(r[1], addr);
    addr = __shfl(addr, (tid + 1) % 4);
    r[2] = __shfl(r[2], addr);
    addr = __shfl(addr, (tid + 1) % 4);
    r[3] = __shfl(r[3], addr);
    // <!-- END: Stage One (Horizontal Rotations) -->


    // <!-- START: Stage Two (Vertical Rotations) -->
    float t[4];
    for (int k = 0; k < 4; ++k)
      t[k] = r[(4 - tid + k) % 4];
    for (int k = 0; k < 4; ++k)
      r[k] = t[k];
    // <!-- END: Stage Two (Vertical Rotations) -->

    // // <!-- START: Stage Two (Vertical Rotations) -->
    // float tmp = r[0];
    // if (tid == 1)
    // {
    //   r[0] = r[3];
    //   r[3] = r[2];
    //   r[2] = r[1];
    //   r[1] = tmp;
    // }
    // else if (tid == 2)
    // {
    //   r[0] = r[2];
    //   r[2] = tmp;
    //   tmp = r[1];
    //   r[1] = r[3];
    //   r[3] = tmp;
    // }
    // else if (tid == 3)
    // {
    //   r[0] = r[1];
    //   r[1] = r[2];
    //   r[2] = r[3];
    //   r[3] = tmp;
    // }
    // // <!-- END: Stage Two (Vertical Rotations) -->

    // <!-- START: Stage Three (Horizontal Rotations) -->
    addr = (4 - tid) % 4;
    r[0] = __shfl(r[0], addr);
    addr = __shfl(addr, (tid + 3) % 4);
    r[1] = __shfl(r[1], addr);
    addr = __shfl(addr, (tid + 3) % 4);
    r[2] = __shfl(r[2], addr);
    addr = __shfl(addr, (tid + 3) % 4);
    r[3] = __shfl(r[3], addr);
    // <!-- END: Stage Three (Horizontal Rotations) -->

    out[col_index] = r[0];
    out[col_index + width * 1] = r[1];
    out[col_index + width * 2] = r[2];
    out[col_index + width * 3] = r[3];

#endif

}


__global__ void transpose_lds_kernel(float *in, float *out, int width,
                                     int height) {
  __shared__ float tile[tile_dim][tile_dim];

  int x_tile_index = blockIdx.x * tile_dim;
  int y_tile_index = blockIdx.y * tile_dim;

  int in_index = (y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
  // int out_index =(y_tile_index + threadIdx.x) * height + (x_tile_index + threadIdx.y);

    tile[threadIdx.y][threadIdx.x] = in[in_index];
    __syncthreads();

    out[in_index] = tile[threadIdx.x][threadIdx.y];
    __syncthreads();
}


int main() {
  std::vector<float> matrix_in;
  std::vector<float> matrix_out;

  matrix_in.resize(width * height * batch);
  matrix_out.resize(width * height * batch);

  for(int b = 0; b < batch; ++b)
  {
    for (int i = 0; i < width * height; i++)
    {
      // matrix_in[(b * width * height) + i] = (float)rand() / (float)RAND_MAX;
      matrix_in[(b * width * height) + i] = i;
    }
  }

#if VERBOSE
  for(int b = 0; b < batch; ++b)
  {
    std::cout << "========= Matrix : " << b << " ============" << std::endl;
    for(size_t i = 0; i < width * height; i++)
    {
        std::cout << /*"[" << i << "]" << */matrix_in[(b * width * height) + i] << "\t";
        if((i + 1) % width == 0)
            std::cout << std::endl;
    }
  }
#endif

  float *d_in;
  float *d_out;

  hipMalloc((void **)&d_in, width * height * batch * sizeof(float));
  hipMalloc((void **)&d_out, width * height * batch * sizeof(float));

  hipMemcpy(d_in, matrix_in.data(), width * height * batch * sizeof(float),
            hipMemcpyHostToDevice);

  printf("Setup complete. Launching kernel \n");
  // int block_x = width / tile_dim;
  // int block_y = height * batch / tile_dim;

#ifdef USE_SHUFFLE
  hipLaunchKernelGGL(transpose_shift_kernel, dim3(1, batch),
                      dim3(tile_dim, 1), 0, 0, d_in, d_out, width,
                      height);
#else
  hipLaunchKernelGGL(transpose_lds_kernel, dim3(1, batch),
                      dim3(tile_dim, tile_dim), 0, 0, d_in, d_out, width,
                      height);
#endif

   hipDeviceSynchronize();

   printf("Kernel execution complete \n");

   hipMemcpy(matrix_out.data(), d_out, width * height * batch * sizeof(float),
            hipMemcpyDeviceToHost);

#if VERBOSE
  for(int b = 0; b < batch; ++b)
  {
    std::cout << "========= Transposed : " << b << " ============" << std::endl;
    for(size_t i = 0; i < width * height; i++)
    {
        std::cout << /*"[" << i << "]" << */matrix_out[(b * width * height) + i] << "\t";
        if((i + 1) % width == 0)
            std::cout << std::endl;
    }
  }
#endif

  return 0;
}