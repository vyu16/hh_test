#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"

/* print matrix in GPU memory */
void print_mat(int nrow, int ncol, int stride, const double* mat)
{
  int i,j;
  double *cpu_mat = (double*)malloc(sizeof(double)*nrow*ncol);
  cublasGetMatrix(nrow,ncol,sizeof(double),(void*)mat,stride,(void*)cpu_mat,nrow);
  for(i=0 ; i<ncol ; i++)
  for(j=0 ; j<nrow ; j++)
  {
    printf("%d %d %e\n",j,i,cpu_mat[j+i*nrow]);
  }
  free(cpu_mat);
  return;
}

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

/* work: ncols*ncols, work2: nev*ncols, work3: (ncols+nb-1)*ncols */
extern "C" void compute_hh_gpu_kernel3(double *q, const double *hh, double *work, double *work2, double *work3, const int nev, const int nb, const int ldq, const int ncols, intptr_t *handle)
{
  int npad = ncols+nb-1;
  cublasStatus_t info;
  double zero = 0.0, half = 0.5, one = 1.0, minus_one = -1.0;

  /* move hh matrix to padded work3 matrix */
  info = cublasDgeam(*((cublasHandle_t*)handle), CUBLAS_OP_N, CUBLAS_OP_N, npad, ncols,	&zero, work3, npad, &zero, work3, npad, work3, npad);
  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dgeam failure\n"); exit(1); }
  info = cublasDgeam(*((cublasHandle_t*)handle), CUBLAS_OP_N, CUBLAS_OP_N, nb, ncols, &one, hh, nb, &zero, hh, nb, work3, npad+1);
  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dgeam failure\n"); exit(1); }

  /* work2 = q*work3 */
  info = cublasDgemm(*((cublasHandle_t*)handle), CUBLAS_OP_N, CUBLAS_OP_N, nev, ncols, npad, &one, q, ldq, work3, npad, &zero, work2, nev);
  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dgemm failure\n"); exit(1); }

  /* work = work3^T*work3 */
  info = cublasDsyrk(*((cublasHandle_t*)handle), CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, ncols, npad, &one, work3, npad, &zero, work, ncols);
  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dsyrk failure\n"); exit(1); }

  /* post-processing of work matrix */
  info = cublasDscal(*((cublasHandle_t*)handle), ncols, &half, work, ncols+1);
  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dscal failure\n"); exit(1); }

  /* work2 = work2*work^{-T} OR work3 = work3*work^{-1} (whichever is more efficient) */
  if( npad < nev )
  { info = cublasDtrsm(*((cublasHandle_t*)handle), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, npad, ncols, &one, work, ncols, work3, npad); }
  else
  { info = cublasDtrsm(*((cublasHandle_t*)handle), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, nev, ncols, &one, work, ncols, work2, nev); }
  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dtrsm failure\n"); exit(1); }

  /* q = q - work2*work3^T */
  info = cublasDgemm(*((cublasHandle_t*)handle), CUBLAS_OP_N, CUBLAS_OP_T, nev, npad, ncols, &minus_one, work2, nev, work3, npad, &one, q, ldq);
  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dgemm failure\n"); exit(1); }

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    printf("\n compute_hh_trafo CUDA kernel #2 failed: %s \n", cudaGetErrorString(err));
    exit(1);
  }
}
