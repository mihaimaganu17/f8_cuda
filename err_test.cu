#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void k()
{ }

#define CUDA_CHECK(expr_to_check) do { \
    cudaError_t result = expr_to_check; \
    if (result != cudaSuccess) { \
        fprintf(stderr, \
            "CUDA Runtime Error: %s.%i:%d = %s\n", \
            __FILE__, \
            __LINE__, \
            result, \
            cudaGetErrorString(result)); \
    } \
} while (0)

int main()
{
        // Check the max TB cluster size
        int clusterSize = 0;
        cudaOccupancyMaxPotentialClusterSize()
        k<<<8192, 4096>>>(); // Invalid block size
        CUDA_CHECK(cudaGetLastError());
        return 0;
}