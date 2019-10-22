module compute_hh_wrapper

  use iso_c_binding, only: c_double,c_intptr_t

  implicit none

  integer :: q_num
  integer :: hh_num
  integer :: tau_num
  integer :: work_num

  integer(c_intptr_t) :: q_dev
  integer(c_intptr_t) :: hh_dev
  integer(c_intptr_t) :: tau_dev
  integer(c_intptr_t) :: work
  integer(c_intptr_t) :: handle

  private :: q_num, hh_num, tau_num
  private :: q_dev, hh_dev, tau_dev

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

  end subroutine

  subroutine free_hh()

    use cuda_f_interface

    implicit none

    integer :: ok

    if(q_num > 0) ok = cuda_free(q_dev)
    if(hh_num > 0) ok = cuda_free(hh_dev)
    if(tau_num > 0) ok = cuda_free(tau_dev)
    if(work_num > 0) ok = cuda_free(work)

    ok = cublas_destroy(handle)

  end subroutine

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
    num = nc*size_of_double
    if(work_num < num) then
      if(work_num > 0) ok = cuda_free(work)
      ok = cuda_malloc(work,num)
      work_num = num
    end if

    ! Compute
    if(version == 1)then
      call compute_hh_gpu_kernel(q_dev,hh_dev,tau_dev,nc,nbw,nc,nn)
    else
      call compute_hh_gpu_kernel2(q_dev,hh_dev,tau,work,nc,nbw,nc,nn,handle)
    endif

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

end module
