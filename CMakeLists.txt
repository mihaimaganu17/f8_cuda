cmake_minimum_required(VERSION 4.3.1)

project(F8Cuda LANGUAGES CXX CUDA)

add_executable(F8Cuda)
target_sources(F8Cuda
    PRIVATE
        vecAdd_unifiedMemory.cu
)
target_compile_options(F8Cuda PRIVATE -Xcompiler "/Zc:preprocessor" -std=c++17)

add_executable(errTest)
target_sources(errTest
    PRIVATE
        err_test.cu
)
target_compile_options(errTest PRIVATE -Xcompiler "/Zc:preprocessor" -std=c++17)

# TODO: Testing with CTest
# Error log management: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#error-log-management
enable_testing()
add_test(
    NAME ErrorWithCUDA_LOG
    COMMAND CUDA_LOG_FILE="cudaLog.txt"; errLog.exe
)