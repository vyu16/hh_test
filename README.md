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
./hh_test b n
```

See `doc.pdf` for the meanings of `b` and `n`. The value of `b` must be one of
32, 64, 128, 256, 512, 1024. The reported error should be a very small value,
otherwise something is wrong.
