#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <vector>

// Permute only helps when exchanging data in one wave.......

const static int width = 4;
const static int height = 4; // single matrix is 16x16
const static int tile_dim = 4;
// const static int batches = 10; // how many 64x64

// row major......
__global__ void transpose_perm_kernel(float *in, float *out, int width,
                                     int height, int batch) {

  // const int x_tile_index = 0;
  // int y_tile_index = blockIdx.y * tile_dim;

  // int in_local_row_index = threadIdx.x * tile_dim;

  // float registers[tile_dim];  // each thread reads a row = tile dim elements

  // // local index in single transpose matrix
  // int trans_local_row_index = threadIdx.x;
  // const int matrixStrides = width * height;
  // while(in_local_row_index < matrixStrides * batch)
  // {
  //   for(size_t elemID = 0; elemID < tile_dim; ++elemID)
  //   {
  //     registers[elemID] = in[in_local_row_index + elemID];
  //   }

  //   for(size_t elemID = threadIdx.x; elemID < tile_dim; ++elemID)
  //   {
  //     float others = __hip_ds_bpermutef(elemID*4, registers[trans_local_row_index]);
  //     __hip_ds_permutef(elemID*4, registers[trans_local_row_index]);
  //     out[in_local_row_index + elemID] = others;
  //   }

  //   for(size_t elemID = 0; elemID < tile_dim; ++elemID)
  //   {
  //     in[in_local_row_index + elemID] = registers[elemID];
  //   }

  //   in_local_row_index += matrixStrides;
  // }


  const int x_tile_index = 0;
  int y_tile_index = blockIdx.y * tile_dim;
  int col_index = (y_tile_index * width) + threadIdx.x; // the start index of this col in this batch

  // float registers[tile_dim];  // each thread reads a column = tile dim elements
  float r[4];  // each thread reads a column = tile dim elements

  // local index in single transpose matrix
  int trans_row_index = threadIdx.x;
  const int matrixStrides = width * height;
  while(col_index < matrixStrides * batch)
  {
    r[0] = in[col_index];
    r[1] = in[col_index + width * 1];
    r[2] = in[col_index + width * 2];
    r[3] = in[col_index + width * 3];

    r[0] = __hip_ds_bpermutef(0, r[threadIdx.x]);
    r[1] = __hip_ds_bpermutef(4, r[threadIdx.x]);
    r[2] = __hip_ds_bpermutef(8, r[threadIdx.x]);
    r[3] = __hip_ds_bpermutef(12, r[threadIdx.x]);

    out[col_index] = r[0];
    out[col_index + width * 1] = r[1];
    out[col_index + width * 2] = r[2];
    out[col_index + width * 3] = r[3];

// #pragma unroll
//     for(size_t elemID = 0; elemID < tile_dim; ++elemID)
//     {
//       registers[elemID] = in[col_index + (width * elemID)];
//       out[col_index + (width * elemID)] = registers[elemID];
//     }

//    if(threadIdx.x == 0)
//    {
// #pragma unroll
//       for(size_t elemID = 0; elemID < tile_dim; ++elemID)
//       {
//         registers[elemID] = __hip_ds_bpermutef(elemID*4, registers[trans_row_index]);
//       }

//       for(size_t elemID = 0; elemID < tile_dim; ++elemID)
//       {
//         out[col_index + (width * elemID)] = registers[elemID];
//       }
//    }
    col_index += matrixStrides;
  }

}


__global__ void transpose_lds_kernel(float *in, float *out, int width,
                                     int height, int batch) {
  __shared__ float tile[tile_dim][tile_dim];

  int x_tile_index = blockIdx.x * tile_dim;
  int y_tile_index = blockIdx.y * tile_dim;

  int in_index = (y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
  int out_index =(x_tile_index + threadIdx.y) * height + (y_tile_index + threadIdx.x);

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
  int batch = 1;

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

  for(int b = 0; b < batch; ++b)
  {
    for(size_t i = 0; i < width * height; i++)
    {
        std::cout << /*"[" << i << "]" << */matrix_in[(b * width * height) + i] << "\t";
        if((i + 1) % width == 0)
            std::cout << std::endl;
    }
    std::cout << "=============" << std::endl;
  }

  float *d_in;
  float *d_out;

  hipMalloc((void **)&d_in, width * height * batch * sizeof(float));
  hipMalloc((void **)&d_out, width * height * batch * sizeof(float));

  hipMemcpy(d_in, matrix_in.data(), width * height * batch * sizeof(float),
            hipMemcpyHostToDevice);

  printf("Setup complete. Launching kernel \n");
  int block_x = width / tile_dim;
  int block_y = (height * batch) / tile_dim;

  hipLaunchKernelGGL(transpose_perm_kernel, dim3(block_x, block_y),
                      dim3(tile_dim, 1), 0, 0, d_in, d_out, width,
                      height, batch);

  // hipLaunchKernelGGL(transpose_lds_kernel, dim3(block_x, block_y),
  //                     dim3(tile_dim, tile_dim), 0, 0, d_in, d_out, width,
  //                     height, batch);

   hipDeviceSynchronize();

   printf("Kernel execution complete \n");

   hipMemcpy(matrix_out.data(), d_out, width * height * batch * sizeof(float),
            hipMemcpyDeviceToHost);


  for(int b = 0; b < batch; ++b)
  {
    for(size_t i = 0; i < width * height; i++)
    {
        std::cout << /*"[" << i << "]" << */matrix_out[(b * width * height) + i] << "\t";
        if((i + 1) % width == 0)
            std::cout << std::endl;
    }
    std::cout << "=============" << std::endl;
  }

  return 0;
}