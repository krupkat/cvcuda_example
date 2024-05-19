cmake_minimum_required(VERSION 3.24)

project(CudaHello LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(cvcuda REQUIRED)
find_package(nvcv_types REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

add_executable(cvcuda_basic
  main.cpp
)

target_link_libraries(cvcuda_basic PRIVATE
  cvcuda
  nvcv_types
  opencv_core
  opencv_imgcodecs
  CUDA::cudart
)