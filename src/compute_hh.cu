/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 Victor Yu
 * Copyright (c) 2020 NVIDIA CORPORATION
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include <cub/cub.cuh>

// Try CUDA 11 warp reduce

#if (CUDART_VERSION >= 9000)
template <typename T, unsigned int blk> __device__ void warp_shfl_reduce_real(volatile T *s_block)
{
    unsigned int tid = threadIdx.x;

    T val;

    if (blk >= 64)
    {
        if (tid < 32)
        {
            s_block[tid] += s_block[tid + 32];
        }
    }

    val = s_block[tid];

    for (int i = 16; i >= 1; i /= 2)
    {
        val += __shfl_xor_sync(0xffffffff, val, i, 32);
    }

    s_block[tid] = val;
}
#endif

template <typename T, unsigned int blk> __device__ void warp_reduce_real(volatile T *s_block)
{
    unsigned int tid = threadIdx.x;

    if (blk >= 64)
    {
        if (tid < 32)
        {
            s_block[tid] += s_block[tid + 32];
        }
    }

    if (blk >= 32)
    {
        if (tid < 16)
        {
            s_block[tid] += s_block[tid + 16];
        }
    }

    if (blk >= 16)
    {
        if (tid < 8)
        {
            s_block[tid] += s_block[tid + 8];
        }
    }

    if (blk >= 8)
    {
        if (tid < 4)
        {
            s_block[tid] += s_block[tid + 4];
        }
    }

    if (blk >= 4)
    {
        if (tid < 2)
        {
            s_block[tid] += s_block[tid + 2];
        }
    }

    if (blk >= 2)
    {
        if (tid < 1)
        {
            s_block[tid] += s_block[tid + 1];
        }
    }
}

template <typename T, unsigned int blk> __device__ void reduce_real(T *s_block)
{
    unsigned int tid = threadIdx.x;

    if (blk >= 1024)
    {
        if (tid < 512)
        {
            s_block[tid] += s_block[tid + 512];
        }

        __syncthreads();
    }

    if (blk >= 512)
    {
        if (tid < 256)
        {
            s_block[tid] += s_block[tid + 256];
        }

        __syncthreads();
    }

    if (blk >= 256)
    {
        if (tid < 128)
        {
            s_block[tid] += s_block[tid + 128];
        }

        __syncthreads();
    }

    if (blk >= 128)
    {
        if (tid < 64)
        {
            s_block[tid] += s_block[tid + 64];
        }

        __syncthreads();
    }

#if (CUDART_VERSION > 9000)
    if (blk >= 32)
    {
        if (tid < 32)
        {
            warp_shfl_reduce_real<T, blk>(s_block);
        }
    }
    else
    {
        if (tid < 32)
        {
            warp_reduce_real<T, blk>(s_block);
        }
    }
#else
    if (tid < 32)
    {
        warp_reduce_real<T, blk>(s_block);
    }
#endif
}

/*
Householder transformation

(I - tau * hh * hh^T) * q = q - tau * hh * hh^T * q

Name here : Name in paper
q         : X
hh        : v
hh_tau    : tau
nev       : N_C
nb        : nbw (==b)
ncols     : N_R (==n+b-1)
*/
template <typename T, unsigned int blk>
__global__ void compute_hh_trafo_kernel_real(T * __restrict__ q, const T * __restrict__ hh, const T * __restrict__ hh_tau, const int nb, const int ldq, const int ncols)
{
    __shared__ T q_s[blk + 1];
    __shared__ T dotp_s[blk];

    T q_v2;

    int q_off, h_off, j;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    j = ncols;
    // q_off bad access!
    q_off = bid + (j + tid - 1) * ldq;
    h_off = tid + (j - 1) * nb;
    q_s[tid] = q[q_off];

    while (j >= 1)
    {
        if (tid == 0)
        {
            q_s[tid] = q[q_off];
        }

        q_v2 = q_s[tid];
        dotp_s[tid] = q_v2 * hh[h_off];

        __syncthreads();

        reduce_real<T, blk>(dotp_s);

        __syncthreads();

        q_v2 -= dotp_s[0] * hh_tau[j - 1] * hh[h_off];
        q_s[tid + 1] = q_v2;

        if ((j == 1) || (tid == blockDim.x - 1))
        {
            q[q_off] = q_v2;
        }

        __syncthreads();

        q_off -= ldq;
        h_off -= nb;
        j -= 1;
    }
}

#define USE_MMA // On Ampere the double precision tensor cores (DMMA) are available

#ifdef USE_MMA
#include "mma_m8n8k4_fp64_sm80.cuh"
#else

template<int bK, int bN>
__device__ inline int shared_memory_offset(int k, int n) {
  // Shared memory layout for non-MMA version.
  return k * bN + n;
}

__device__ inline constexpr int shared_memory_bytes(int bK, int bN) {
  // Shared memory size for the bM by bK matrix. Version for the non-MMA.
  return bN * bK;
}

#endif

/*
Householder transformation

This is based the on the original warp sync version shown above.

(I - tau * hh * hh^T) * q = q - tau * hh * hh^T * q

Name here : Name in paper
q         : X
hh        : v
hh_tau    : tau
nev       : N_C
nb        : nbw (==b)
ncols     : N_R (==n+b-1)
*/
template <typename T, int bM, int bN, int block_y, int block_z>
__global__ void compute_hh_trafo_gpu(T * __restrict__ q, const T * __restrict__ hh, const T * __restrict__ hh_tau, const int nev, const int nb, const int ldq, const int ncols)
{
  constexpr int bK = bM;

  extern __shared__ int smem[];

  T *hh_s = reinterpret_cast<T *>(smem);
  T *q_s = &hh_s[bM];
  T *hh_tau_s = &q_s[shared_memory_bytes(bK, bN)];
#ifdef USE_MMA
  T *sum_s = &hh_tau_s[1]; // Shared memory buffer if we perform the inner product with DMMA.
#endif

  int j = ncols;

  int bid = blockIdx.y * bN; // n-index offset for this block.

  for (int k = threadIdx.z; k < bK; k += block_z) {
    for (int n = threadIdx.y; n < bN; n += block_y) {
      q_s[shared_memory_offset<bK, bN>(k, n)] = (n + bid) < nev ? q[(j + k - 1) * ldq + n + bid] : 0;
    }
  }

  constexpr int thread_m_dim = bM / block_z;
  constexpr int thread_n_dim = bN / block_y;

  T reg[thread_n_dim * thread_m_dim];

  while (j >= 1)
  {
    int hh_idx = threadIdx.z * blockDim.y + threadIdx.y;
    if (hh_idx == 0) { *hh_tau_s = hh_tau[j - 1]; }
    while (hh_idx < nb) {
      hh_s[hh_idx] = hh[hh_idx + (j - 1) * nb];
      hh_idx += blockDim.z * blockDim.y;
    }

    if (j < ncols && threadIdx.z == 0) {
      for (int n = threadIdx.y; n < bN; n += block_y) {
        q_s[shared_memory_offset<bK, bN>(0, n)] = (n + bid) < nev ? q[(j + 0 - 1) * ldq + n + bid] : 0;
      }
    }

/**
  If we use DMMA to perform the inner product, call the routine here and store results on the buffer.
  If not, for each eigenvector, for each thread we calculate the `sum`.
 */

#ifdef USE_MMA
    __syncthreads();
    sum<bK, bN, block_z * block_y / 32>(hh_s, q_s, sum_s);
    __syncthreads();
#endif

#pragma unroll
    for (int n = 0; n < thread_n_dim; n++) {
      int n_idx = threadIdx.y + n * block_y;

#ifndef USE_MMA
    T sum = 0;
#pragma unroll 1
    for (int k = 0; k < bK; k++) {
      sum += hh_s[k] * q_s[shared_memory_offset<bK, bN>(k, n_idx)];
    }
#endif

#pragma unroll
      for (int m = 0; m < thread_m_dim; m++) {
        int m_idx = threadIdx.z + m * block_z;
#ifdef USE_MMA
        reg[m * thread_n_dim + n] = q_s[shared_memory_offset<bK, bN>(m_idx, n_idx)] - *hh_tau_s * hh_s[m_idx] * sum_s[n_idx];
#else
        reg[m * thread_n_dim + n] = q_s[shared_memory_offset<bK, bN>(m_idx, n_idx)] - *hh_tau_s * hh_s[m_idx] * sum;
#endif
        if (j == 1 || m_idx == bM - 1) {
          if (n_idx + bid < nev) { q[(m_idx + j - 1) * ldq + n_idx + bid] = reg[m * thread_n_dim + n]; }
        }
      }
    }

    __syncthreads();

#pragma unroll
    for (int m = 0; m < thread_m_dim; m++) {
#pragma unroll
      for (int n = 0; n < thread_n_dim; n++) {
        int m_idx = threadIdx.z + m * block_z;
        int n_idx = threadIdx.y + n * block_y;
        if (m_idx + 1 < bM) { q_s[shared_memory_offset<bK, bN>(m_idx + 1, n_idx)] = reg[m * thread_n_dim + n]; }
      }
    }

    j -= 1;
  }
}

void set_max_shared_bytes(const void *func)
{
  // Set such that this kernel can use the maximum shared memory available.
  cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared);
  int max_shared_bytes;
  cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes);
}

template <int bM, class F>
void launch_new_kernel(F *q, const F *hh, const F *hh_tau, const int nev, const int nb, const int ldq, const int ncols)
{
#ifdef USE_MMA
  // This is set such that shared memory bank conflicts are minimized.
  constexpr int block_y = bM < 64 ? 8 : 4;
  constexpr int block_z = bM < 64 ? 4 : 8;
#else
  constexpr int block_y = 8;
  constexpr int block_z = 4;
#endif
  constexpr int bN = 8;
  auto kernel = compute_hh_trafo_gpu<double, bM, bN, block_y, block_z>;
  set_max_shared_bytes((const void *)kernel);
#ifdef USE_MMA
  int shared_bytes = (bM + shared_memory_bytes(bM, bN) + bN + 1) * sizeof(F);
#else
  int shared_bytes = (bM + shared_memory_bytes(bM, bN) + 1) * sizeof(F);
#endif
  int grid_y = (nev + bN - 1) / bN;
  kernel<<<dim3(1, grid_y, 1), dim3(1, block_y, block_z), shared_bytes>>>(q, hh, hh_tau, nev, nb, ldq, ncols);
}

/*
Name here : Name in paper
q         : X
hh        : v
hh_tau    : tau
nev       : N_C
nb        : nbw (==b)
ncols     : N_R (==n+b-1)
*/
extern "C" void compute_hh_gpu_kernel(double *q, const double *hh, const double *hh_tau, const int nev, const int nb, const int ldq, const int ncols)
{

    cudaEvent_t new_start;
    cudaEvent_t new_stop;

    cudaEvent_t old_start;
    cudaEvent_t old_stop;

    cudaEventCreate(&new_start);
    cudaEventCreate(&new_stop);

    cudaEventCreate(&old_start);
    cudaEventCreate(&old_stop);

    // Create duplicate device buffers for the old kernel.
    double *q_x;
    cudaMalloc(&q_x, (ncols + nb - 1) * ldq * sizeof(double));
    cudaMemcpy(q_x, q, (ncols + nb - 1) * ldq, cudaMemcpyDeviceToDevice);
    double *hh_x;
    cudaMalloc(&hh_x, ncols * nb * sizeof(double));
    cudaMemcpy(hh_x, hh, ncols * nb, cudaMemcpyDeviceToDevice);
    double *hh_tau_x;
    cudaMalloc(&hh_tau_x, ncols * sizeof(double));
    cudaMemcpy(hh_tau_x, hh_tau, ncols, cudaMemcpyDeviceToDevice);

    // Run the old kernel
    cudaEventRecord(old_start);

    switch (nb)
    {
      case 1024:
        compute_hh_trafo_kernel_real<double, 1024><<<nev, nb>>>(q_x, hh_x, hh_tau_x, nb, ldq, ncols);
        break;
      case 512:
        compute_hh_trafo_kernel_real<double, 512><<<nev, nb>>>(q_x, hh_x, hh_tau_x, nb, ldq, ncols);
        break;
      case 256:
        compute_hh_trafo_kernel_real<double, 256><<<nev, nb>>>(q_x, hh_x, hh_tau_x, nb, ldq, ncols);
        break;
      case 128:
        compute_hh_trafo_kernel_real<double, 128><<<nev, nb>>>(q_x, hh_x, hh_tau_x, nb, ldq, ncols);
        break;
      case 64:
        compute_hh_trafo_kernel_real<double, 64><<<nev, nb>>>(q_x, hh_x, hh_tau_x, nb, ldq, ncols);
        break;
      case 32:
        compute_hh_trafo_kernel_real<double, 32><<<nev, nb>>>(q_x, hh_x, hh_tau_x, nb, ldq, ncols);
        break;
      case 16:
        compute_hh_trafo_kernel_real<double, 16><<<nev, nb>>>(q_x, hh_x, hh_tau_x, nb, ldq, ncols);
        break;
      case 8:
        compute_hh_trafo_kernel_real<double, 8><<<nev, nb>>>(q_x, hh_x, hh_tau_x, nb, ldq, ncols);
        break;
      case 4:
        compute_hh_trafo_kernel_real<double, 4><<<nev, nb>>>(q_x, hh_x, hh_tau_x, nb, ldq, ncols);
        break;
      case 2:
        compute_hh_trafo_kernel_real<double, 2><<<nev, nb>>>(q_x, hh_x, hh_tau_x, nb, ldq, ncols);
        break;
      case 1:
        compute_hh_trafo_kernel_real<double, 1><<<nev, nb>>>(q_x, hh_x, hh_tau_x, nb, ldq, ncols);
        break;
      default: printf("Unsupported nb = %d", nb);
    }

    cudaEventRecord(old_stop);
    cudaEventSynchronize(old_stop);

    cudaFree(q_x);
    cudaFree(hh_x);
    cudaFree(hh_tau_x);

    // Run the new kernel
    cudaEventRecord(new_start);

    switch (nb) {
      case 1024: launch_new_kernel<1024>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
      case  512: launch_new_kernel< 512>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
      case  256: launch_new_kernel< 256>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
      case  128: launch_new_kernel< 128>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
      case   64: launch_new_kernel<  64>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
      case   32: launch_new_kernel<  32>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
      case   16: launch_new_kernel<  16>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
      case    8: launch_new_kernel<   8>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
      default: printf("Unsupported nb = %d", nb);
    }

    cudaEventRecord(new_stop);
    cudaEventSynchronize(new_stop);

    // Collect time and perf
    float old_time;
    float new_time;
    cudaEventElapsedTime(&new_time, new_start, new_stop);
    cudaEventElapsedTime(&old_time, old_start, old_stop);
    double ops = nev * nb * (2.0 + 3.0) * ncols;
    double bytes = ((nb * ncols) + (nb + ncols - 1) * nev * 2.0) * sizeof(double);
    printf("Old kernel took %8.4f ms, GFLOPS = %6.1f, GBS = %6.1f\n", old_time, ops / (old_time * 1e-3) / 1e+9, bytes / (old_time * 1e-3) / 1e+9);
    printf("New kernel took %8.4f ms, GFLOPS = %6.1f, GBS = %6.1f\n", new_time, ops / (new_time * 1e-3) / 1e+9, bytes / (new_time * 1e-3) / 1e+9);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("\n compute_hh_trafo CUDA kernel failed: %s \n", cudaGetErrorString(err));
        exit(1);
    }
}
