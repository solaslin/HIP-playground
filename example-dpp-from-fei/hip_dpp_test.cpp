#include <hip/hip_runtime.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>

#define TRIAL_NUM 101

#define HIP_CHECK(condition)                                                           \
    {                                                                                  \
        hipError_t error = condition;                                                  \
        if(error != hipSuccess)                                                        \
        {                                                                              \
            std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
            exit(error);                                                               \
        }                                                                              \
    }

using T = int;

template <class T, int dpp_ctrl, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = false>
__device__ inline T warp_move_dpp(T input)
{
    constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    int words[words_no];
    __builtin_memcpy(words, &input, sizeof(T));

#pragma unroll
    for(int i = 0; i < words_no; i++)
    {
        words[i] = ::__builtin_amdgcn_mov_dpp(words[i], dpp_ctrl, row_mask, bank_mask, bound_ctrl);
    }

    T output;
    __builtin_memcpy(&output, words, sizeof(T));

    return output;
}

template <class T, int pattern>
__device__ inline T warp_ds_swizle(T input)
{
    constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    int words[words_no];
    __builtin_memcpy(words, &input, sizeof(T));

#pragma unroll
    for(int i = 0; i < words_no; i++)
    {
        words[i] = ::__builtin_amdgcn_ds_swizzle(words[i], pattern);
    }

    T output;
    __builtin_memcpy(&output, words, sizeof(T));

    return output;
}

__global__ void ds_rw(const T* d_input, T* d_output)
{
    extern __shared__ T lds[];
    int                 idx = threadIdx.x + blockIdx.x * blockDim.x;

    lds[threadIdx.x] = d_input[idx];

    __syncthreads();

    T tmp;
#pragma unroll
    for(int i = 0; i < TRIAL_NUM; i++)
    {
        tmp = lds[threadIdx.x ^ 0x1];
        __syncthreads();
        tmp++;
        lds[threadIdx.x] = tmp;
        __syncthreads();
    }

    d_output[idx] = tmp;
}

__global__ void ds_swizzle(const T* d_input, T* d_output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // d_output[idx] = warp_ds_swizle<T, 0x41F>(d_input[idx]);
    T tmp = d_input[idx];
#pragma unroll
    for(int i = 0; i < TRIAL_NUM; i++)
    {
        tmp = __hip_ds_swizzle(tmp, 0x41F);
        tmp++;
    }
    d_output[idx] = tmp;
}

__global__ void dpp_quad_perm(const T* d_input, T* d_output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // quad_perm:[1,0,3,2] -> 10110001
    T tmp = d_input[idx];
#pragma unroll
    for(int i = 0; i < TRIAL_NUM; i++)
    {
        tmp = warp_move_dpp<T, 0xb1>(tmp);
        tmp++;
    }
    d_output[idx] = tmp;
}

int main(int argc, char** argv)
{
    int test_id = 0;

    if(argc >= 1)
        test_id = atoi(argv[1]);

    const int blocks = 64;
    const int grid   = 64;
    const int size   = grid * blocks;

    std::vector<T> input(size);
    std::iota(input.begin(), input.end(), 0);
    std::vector<T> output(size, -1);

    T* d_input;
    T* d_output;

    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));

    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));

    switch(test_id)
    {
    case 0:
        ds_rw<<<dim3(grid), dim3(blocks), blocks * sizeof(T)>>>(d_input, d_output);
        break;
    case 1:
        ds_swizzle<<<dim3(grid), dim3(blocks)>>>(d_input, d_output);
        break;
    case 2:
        dpp_quad_perm<<<dim3(grid), dim3(blocks)>>>(d_input, d_output);
        break;
    }

    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output.data(), d_output, size * sizeof(T), hipMemcpyDeviceToHost));

    for(size_t i = 0; i < size; i++)
    {
        std::cout << "[" << i << "]" << output[i] << "\t";
    }
    std::cout << std::endl;

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    return 0;
}
