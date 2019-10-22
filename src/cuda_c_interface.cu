#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"

extern "C" {

int cuda_set_device(int i_gpu)
{
    cudaError_t err = cudaSetDevice(i_gpu);

    if (err != cudaSuccess)
    {
        printf("\n Error in cudaSetDevice: %s \n", cudaGetErrorString(err));
        exit(1);
    }

    return 0;
}

int cuda_get_device_count(int *n_gpu)
{
    cudaError_t err = cudaGetDeviceCount(n_gpu);

    if (err != cudaSuccess)
    {
        printf("\n Error in cudaGetDeviceCount: %s \n", cudaGetErrorString(err));
        exit(1);
    }

    return 0;
}

int cuda_device_synchronize()
{
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("\n Error in cudaDeviceSynchronize: %s \n", cudaGetErrorString(err));
    }

    return 0;
}

int cuda_malloc(intptr_t *a, size_t size)
{
    cudaError_t err = cudaMalloc((void **) a, size);

    if (err != cudaSuccess)
    {
        printf("\n Error in cudaMalloc: %s \n", cudaGetErrorString(err));
        exit(1);
    }

    return 0;
}

int cuda_free(intptr_t *a)
{
    cudaError_t err = cudaFree(a);

    if (err != cudaSuccess)
    {
        printf("\n Error in cudaFree: %s \n", cudaGetErrorString(err));
        exit(1);
    }

    return 0;
}

int cuda_memcpy(intptr_t *dest, intptr_t *src, size_t count, int dir)
{
    cudaMemcpyKind dir2;

    switch (dir)
    {
        case 0:
            dir2 = cudaMemcpyHostToDevice;
            break;
        case 1:
            dir2 = cudaMemcpyDeviceToHost;
            break;
        case 2:
            dir2 = cudaMemcpyDeviceToDevice;
            break;
    }

    cudaError_t err = cudaMemcpy(dest, src, count, dir2);

    if (err != cudaSuccess)
    {
        printf("\n Error in cudaMemcpy: %s \n", cudaGetErrorString(err));
        exit(1);
    }

    return 0;
}

int cublas_create(intptr_t *cublas_handle)
{
    *cublas_handle = (intptr_t) malloc(sizeof(cublasHandle_t));
    cublasStatus_t status = cublasCreate((cublasHandle_t*) *cublas_handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error in cublasCreate\n");
        exit(1);
    }

    return 0;
}

int cublas_destroy(intptr_t *cublas_handle)
{
    cublasStatus_t status = cublasDestroy(*((cublasHandle_t*) *cublas_handle));
    *cublas_handle = (intptr_t) NULL;

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error in cublasDestory\n");
        exit(1);
    }

    return 0;
}

int cublas_set_stream(intptr_t *cublas_handle, intptr_t *cuda_stream)
{
    cublasStatus_t status = cublasSetStream(*((cublasHandle_t*) *cublas_handle), *((cudaStream_t*) *cuda_stream));

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error in cublasSetStream\n");
        exit(1);
    }

    return 0;
}

void cublas_dgemm(intptr_t handle, char transa, char transb, int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc)
{
    cublasDgemm(*((cublasHandle_t*)handle), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

}
