module compute_hh_wrapper

  use iso_c_binding, only: c_double,c_intptr_t

  implicit none

  integer :: q_num
  integer :: hh_num
  integer :: tau_num
  integer :: work_num
  integer :: work2_num
  integer :: work3_num

  integer(c_intptr_t) :: handle
  integer(c_intptr_t) :: q_dev
  integer(c_intptr_t) :: hh_dev
  integer(c_intptr_t) :: tau_dev
  integer(c_intptr_t) :: work_dev
  integer(c_intptr_t) :: work2_dev
  integer(c_intptr_t) :: work3_dev

  private :: q_num, hh_num, tau_num, work_num, work2_num, work3_num
  private :: q_dev, hh_dev, tau_dev, work_dev, work2_dev, work3_dev

  contains

  subroutine init_hh()

    use cuda_f_interface

    implicit none

    integer :: ok

    call gpu_init()

    ok = cublas_create(handle)

    q_num = 0
    hh_num = 0
    tau_num = 0
    work_num = 0
    work2_num = 0
    work3_num = 0

  end subroutine

  subroutine free_hh()

    use cuda_f_interface

    implicit none

    integer :: ok

    if(q_num > 0) ok = cuda_free(q_dev)
    if(hh_num > 0) ok = cuda_free(hh_dev)
    if(tau_num > 0) ok = cuda_free(tau_dev)
    if(work_num > 0) ok = cuda_free(work_dev)
    if(work2_num > 0) ok = cuda_free(work2_dev)
    if(work3_num > 0) ok = cuda_free(work3_dev)

    ok = cublas_destroy(handle)

  end subroutine

  ! version = 1 (CUDA BLAS level-1 kernel)
  ! version = 2 (cuBLAS level-2 kernel)
  subroutine compute_hh_gpu(nn,nc,nbw,q,hh,tau,version)

    use cuda_f_interface

    implicit none

    integer, intent(in) :: nn ! (n)
    integer, intent(in) :: nc ! (N_C)
    integer, intent(in) :: nbw ! (b)
    integer, intent(in) :: version
    real(c_double), intent(inout) :: q(:,:) ! (X), dimension(nc,nn+nbw-1)
    real(c_double), intent(in) :: hh(:,:) ! (v), dimension(nbw,nn)
    real(c_double), intent(in) :: tau(:) ! (tau), dimension(nn)

    integer :: ok
    integer(c_intptr_t) :: num
    integer(c_intptr_t) :: host_ptr

    integer, parameter :: size_of_double = 8

    ! Copy q to GPU
    num = nc*(nn+nbw-1)*size_of_double
    host_ptr = int(loc(q),c_intptr_t)
    if(q_num < num) then
      if(q_num > 0) ok = cuda_free(q_dev)
      ok = cuda_malloc(q_dev,num)
      q_num = num
    end if
    ok = cuda_memcpy(q_dev,host_ptr,num,cudaMemcpyHostToDevice)

    ! Copy hh to GPU
    num = nbw*nn*size_of_double
    host_ptr = int(loc(hh),c_intptr_t)
    if(hh_num < num) then
      if(hh_num > 0) ok = cuda_free(hh_dev)
      ok = cuda_malloc(hh_dev,num)
      hh_num = num
    end if
    ok = cuda_memcpy(hh_dev,host_ptr,num,cudaMemcpyHostToDevice)

    ! Copy tau to GPU
    num = nn*size_of_double
    host_ptr = int(loc(tau),c_intptr_t)
    if(tau_num < num) then
      if(tau_num > 0) ok = cuda_free(tau_dev)
      ok = cuda_malloc(tau_dev,num)
      tau_num = num
    end if
    if(version == 1) then
      ok = cuda_memcpy(tau_dev,host_ptr,num,cudaMemcpyHostToDevice)
    endif

    ! Allocate new workspace if necessary
    num = nn*nn*size_of_double
    if(work_num < num) then
      if(work_num > 0) ok = cuda_free(work_dev)
      ok = cuda_malloc(work_dev,num)
      work_num = num
    end if

    num = nc*nn*size_of_double
    if(work2_num < num) then
      if(work2_num > 0) ok = cuda_free(work2_dev)
      ok = cuda_malloc(work2_dev,num)
      work2_num = num
    end if

    num = (nn+nbw-1)*nn*size_of_double
    if(work3_num < num) then
      if(work3_num > 0) ok = cuda_free(work3_dev)
      ok = cuda_malloc(work3_dev,num)
      work3_num = num
    end if

    ! Compute
    if(version == 1) then
      call compute_hh_gpu_kernel(q_dev,hh_dev,tau_dev,nc,nbw,nc,nn)
    else if(version == 2) then
      call compute_hh_gpu_kernel2(q_dev,hh_dev,tau,work2_dev,nc,nbw,nc,nn,handle)
    else if(version == 3) then
      call compute_hh_gpu_kernel3(q_dev,hh_dev,work_dev,work2_dev,work3_dev,nc,nbw,nc,nn,handle)
    end if

    ! Copy q to CPU
    num = nc*(nn+nbw-1)*size_of_double
    host_ptr = int(loc(q),c_intptr_t)
    ok = cuda_memcpy(host_ptr,q_dev,num,cudaMemcpyDeviceToHost)

  end subroutine

  ! Householder transformation
  ! (I - tau * hh * hh^T) * q = q - tau * hh * hh^T * q
  subroutine compute_hh_cpu(nn,nc,nbw,q,hh,tau)

    implicit none

    integer, intent(in) :: nn ! (n)
    integer, intent(in) :: nc ! (N_C)
    integer, intent(in) :: nbw ! (b)
    real(c_double), intent(inout) :: q(:,:) ! (X), dimension(nc,nn+nbw-1)
    real(c_double), intent(in) :: hh(:,:) ! (v), dimension(nbw,nn)
    real(c_double), intent(in) :: tau(:) ! (tau), dimension(nn)

    integer :: j
    integer :: i
    real(c_double) :: dotp

    do j = nn,1,-1
      do i = 1,nc
        dotp = dot_product(q(i,j:j+nbw-1),hh(:,j))
        q(i,j:j+nbw-1) = q(i,j:j+nbw-1)-tau(j)*dotp*hh(:,j)
      end do
    end do

  end subroutine

  subroutine compute_hh_cpu2(nn,nc,nbw,q,hh,work,work2,work3)

    implicit none

    integer, intent(in) :: nn ! (n)
    integer, intent(in) :: nc ! (N_C)
    integer, intent(in) :: nbw ! (b)

    real(c_double), intent(inout) :: q(:,:) ! (X), dimension(nc,nn+nbw-1)
    real(c_double), intent(in) :: hh(:,:) ! (v), dimension(nbw,nn)

    real(c_double), intent(out) :: work(:,:)
    real(c_double), intent(out) :: work2(:,:)
    real(c_double), intent(out) :: work3(:,:)

    integer :: npad

    ! move hh matrix to padded work3 matrix
    npad = nn + nbw - 1
    call dlaset('A', npad, nn, 0.0, 0.0, work3, npad)
    call dlacpy('A', nbw,  nn, hh, nbw, work3, npad+1)

    ! work2 = q*work3
    call dgemm('N', 'N', nc, nn, npad, 1.0d0, q, nc, work3, npad, 0.0d0, work2, nc)
    ! work = work3^T*work3
    call dsyrk('U', 'T', nn, npad, 1.0d0, work3, npad, 0.0d0, work, nn)
    ! post-processing of work matrix
    call dscal(nn, 0.5d0, work, nn+1)
    ! work2 = work2*work^{-T} OR work3 = work3*work^{-1} (whichever is more efficient)
    if( npad < nc ) then
      call dtrsm('R', 'U', 'N', 'N', npad, nn, 1.0d0, work, nn, work3, npad)
    else
      call dtrsm('R', 'U', 'T', 'N', nc, nn, 1.0d0, work, nn, work2, nc)
    end if
    ! q = q - work2*work3^T
    call dgemm('N', 'T', nc, npad, nn, -1.0d0, work2, nc, work3, npad, 1.0d0, q, nc)

  end subroutine

!/* work: ncols*ncols, work2: nev*ncols, work3: (ncols+nb-1)*ncols */
!extern "C" void compute_hh_gpu_kernel3(double *q, const double *hh, double *work, double *work2, double *work3, const int nev, const int nb, const int ldq, const int ncols, intptr_t *handle)
!{
!  int npad = ncols+nb-1;
!  cublasStatus_t info;
!  double zero = 0.0, half = 0.5, one = 1.0, minus_one = -1.0;
!
!  /* move hh matrix to padded work3 matrix */
!  info = cublasDgeam(*((cublasHandle_t*)handle), CUBLAS_OP_N, CUBLAS_OP_N, npad, ncols, &zero, work3, npad, &zero, work3, npad, work3, npad);
!  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dgeam failure\n"); exit(1); }
!  info = cublasDgeam(*((cublasHandle_t*)handle), CUBLAS_OP_N, CUBLAS_OP_N, nb, ncols, &one, hh, nb, &zero, hh, nb, work3, npad+1);
!  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dgeam failure\n"); exit(1); }
!
!  /* work2 = q*work3 */
!  info = cublasDgemm(*((cublasHandle_t*)handle), CUBLAS_OP_N, CUBLAS_OP_N, nev, ncols, npad, &one, q, ldq, work3, npad, &zero, work2, nev);
!  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dgemm failure\n"); exit(1); }
!
!  /* work = work3^T*work3 */
!  info = cublasDsyrk(*((cublasHandle_t*)handle), CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, ncols, npad, &one, work3, npad, &zero, work, ncols);
!  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dsyrk failure\n"); exit(1); }
!
!  /* post-processing of work matrix */
!  info = cublasDscal(*((cublasHandle_t*)handle), ncols, &half, work, ncols+1);
!  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dscal failure\n"); exit(1); }
!
!  /* work2 = work2*work^{-T} OR work3 = work3*work^{-1} (whichever is more efficient) */
!  if( npad < nev )
!  { info = cublasDtrsm(*((cublasHandle_t*)handle), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, npad, ncols, &one, work, ncols, work3, npad); }
!  else
!  { info = cublasDtrsm(*((cublasHandle_t*)handle), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, nev, ncols, &one, work, ncols, work2, nev); }
!  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dtrsm failure\n"); exit(1); }
!
!  /* q = q - work2*work3^T */
!  info = cublasDgemm(*((cublasHandle_t*)handle), CUBLAS_OP_N, CUBLAS_OP_T, nev, npad, ncols, &minus_one, work2, nev, work3, npad, &one, q, ldq);
!  if(info != CUBLAS_STATUS_SUCCESS) { printf("ERROR: Dgemm failure\n"); exit(1); }

end module
