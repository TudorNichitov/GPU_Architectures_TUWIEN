cmake_minimum_required(VERSION 3.16)
project(ass2 CUDA)


set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
endif ()

message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}. Use -DCMAKE_BUILD_TYPE=<type> (Release/Debug) to change.")
message("Compiler flags: ${CMAKE_CXX_FLAGS}")

add_executable(ass2 src/main.cu)

set_target_properties(ass2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES OFF)
