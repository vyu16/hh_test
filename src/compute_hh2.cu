#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"

/*
Name here : Name in doc
q         : X
hh        : v
hh_tau    : tau
nev       : N_C
nb        : nbw (==b)
ncols     : N_R (==n+b-1)
*/
extern "C" void compute_hh_gpu_kernel2(double *q, const double *hh, const double *hh_tau, double *work, const int nev, const int nb, const int ldq, const int ncols, intptr_t *handle)
{
  int i;
  cublasStatus_t info;
  double zero = 0.0, minus_one = -1.0;

  /* loop over sparse Householder vectors */
  for(i=ncols-1 ; i>=0 ; i--)
  {
    /* apply sparse Householder vector to eigenvectors */
    info = cublasDgemv(*((cublasHandle_t*)handle), CUBLAS_OP_N, nev, nb, hh_tau+i, q+i*ldq, ldq, hh+i*nb, 1, &zero, work, 1);
    if(info != CUBLAS_STATUS_SUCCESS)
    { printf("ERROR: Dgemv failure\n"); exit(1); }

    /* apply rank-1 update to eigenvectors */
    info = cublasDger(*((cublasHandle_t*)handle), nev, nb, &minus_one, work, 1, hh+i*nb, 1, q+i*ldq, ldq);
    if(info != CUBLAS_STATUS_SUCCESS)
    { printf("ERROR: Dger failure\n"); exit(1); }
  }

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    printf("\n compute_hh_trafo CUDA kernel #2 failed: %s \n", cudaGetErrorString(err));
    exit(1);
  }
}
