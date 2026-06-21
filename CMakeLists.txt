cmake_minimum_required(VERSION 4.3.1)

project(F8Cuda LANGUAGES CXX CUDA)

add_executable(F8Cuda)
target_sources(F8Cuda
    PRIVATE
        vecAdd_unifiedMemory.cu
)
target_compile_options(F8Cuda PRIVATE -Xcompiler "/Zc:preprocessor" -std=c++17)
