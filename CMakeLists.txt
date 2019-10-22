cmake_minimum_required(VERSION 3.8)

project(hh_test VERSION 0.0.1 LANGUAGES Fortran CUDA)

#enable_language(Fortran)
#enable_language(CUDA)

add_executable(hh_test
src/compute_hh.cu
src/compute_hh2.cu
src/compute_hh_wrapper.f90
src/cuda_c_interface.cu
src/cuda_f_interface.f90
src/hh_test.f90)

target_link_libraries(hh_test cublas)

#set_target_properties(hh_test PROPERTIES LINKER_LANGUAGE Fortran)
#set_target_properties(hh_test PROPERTIES
#  Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)

# See gitlab.kitware.com/cmake/cmake/issues/17797
if (NOT CMAKE_GENERATOR STREQUAL Ninja AND ${CMAKE_VERSION} VERSION_LESS 3.11)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
endif()
