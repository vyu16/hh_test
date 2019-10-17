## About

Householder transformation mini-app for OLCF GPU Hackathon (October 2019).

## Install

* CMake (3.8+)
* Fortran compiler
* CUDA

```
mkdir build
cd build
cmake ..
```

Explicitly define Fortran compiler and CUDA installation path if necessary.

## Run

```
./hh_test arg1 arg2
```

* arg1: Length of an individual Householder vector. Must be 2^n, n = 5,6,...,10
* arg2: Length of eigenvectors. In real calculations it is usually 100 ~ 1000

The reported error should be a small value, otherwise something is wrong.
