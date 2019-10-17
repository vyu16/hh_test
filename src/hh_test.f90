!
! This program tests the CUDA kernel for Householder transformations.
! It is separeted from the 4th step of the ELPA2 eigensolver.
! Some variables are renamed as follows:
!
! Name in ELPA2 | Name here
! -------------------------
! nbw           | nbw
! max_blk_size  | na
! a_dim2        | na+nbw
! stripe_width  | stripe
! stripe_count  | 1
! istripe       | 1
! off           | 0
! a_off         | 0
! nl            | stripe
! ncols         | na
!
program hh_test

  use iso_c_binding, only: c_double
  use compute_hh_wrapper, only: compute_hh_cpu,compute_hh_gpu

  implicit none

  character(10) :: arg

  integer :: nbw
  integer :: na
  integer :: n

  integer :: i
  real(c_double) :: dotp
  real(c_double) :: err

  integer, allocatable :: seed(:)

  real(c_double), allocatable :: evec1(:,:)
  real(c_double), allocatable :: evec2(:,:)
  real(c_double), allocatable :: hh(:,:)
  real(c_double), allocatable :: tau(:)

  integer, parameter :: stripe = 1024

  ! Read command line arguments
  if(command_argument_count() == 2) then
    call get_command_argument(1,arg)

    read(arg,*) nbw

    ! Must be 2^n
    if(nbw <= 32) then
      nbw = 32
    else if(nbw <= 64) then
      nbw = 64
    else if(nbw <= 128) then
      nbw = 128
    else if(nbw <= 256) then
      nbw = 256
    else if(nbw <= 512) then
      nbw = 512
    else
      nbw = 1024
    end if

    call get_command_argument(2,arg)

    read(arg,*) na

    if(na <= 0) then
      na = 1000
    end if

    write(*,"(2X,A)") "Test parameters:"
    write(*,"(2X,A,I10)") "| nbw ",nbw
    write(*,"(2X,A,I10)") "| na  ",na
  else
    write(*,"(2X,A)") "################################################"
    write(*,"(2X,A)") "##  Wrong number of command line arguments!!  ##"
    write(*,"(2X,A)") "##  Arg#1: Length of Householder vector       ##"
    write(*,"(2X,A)") "##         (must be 2^n, n = 5,6,...,10)      ##"
    write(*,"(2X,A)") "##  Arg#2: Length of eigenvectors             ##"
    write(*,"(2X,A)") "################################################"

    stop
  end if

  ! Generate random data
  call random_seed(size=n)

  allocate(seed(n))
  allocate(evec1(stripe,na+nbw))
  allocate(evec2(stripe,na+nbw))
  allocate(hh(nbw,na))
  allocate(tau(na))

  seed = 20191015

  call random_seed(put=seed)
  call random_number(hh)
  call random_number(evec1)

  ! Normalize
  do i = 1,stripe
    dotp = dot_product(evec1(i,:),evec1(i,:))
    evec1(i,:) = evec1(i,:)/sqrt(abs(dotp))
  end do

  do i = 1,na
    dotp = dot_product(hh(:,i),hh(:,i))
    hh(:,i) = hh(:,i)/sqrt(abs(dotp))
  end do

  evec2 = evec1
  tau = hh(1,:)
  hh(1,:) = 1.0

  ! Start testing CPU reference code
  call compute_hh_cpu(na,stripe,nbw,evec1,hh,tau)

  write(*,"(2X,A)") "CPU version finished"

  ! Start testing GPU code
  call compute_hh_gpu(na,stripe,nbw,evec2,hh,tau)

  write(*,"(2X,A)") "GPU version finished"

  ! Compare results
  err = maxval(abs(evec1-evec2))

  write(*,"(2X,A,E10.2)") "| Error :",err

  deallocate(seed)
  deallocate(evec1)
  deallocate(evec2)
  deallocate(hh)
  deallocate(tau)

end program
