#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>
// __global__ specifier indicates to the compiler that this function will be allowed to be launched
// as a kernel for the GPU
// A kernel launch usually starts from the CPU and return void
__global__ void vecAdd(float* A, float* B, float* C, int vecLen)
{
    // Perform element-wise addition of `A` and `B` and store results in `C`
    // Kernel is parallelized such that each thread will perform one addition. Which element it
    // computes is determined by its thread and grid index.
    int workIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Bounds checking we are not exceeding vector length
    if (workIndex < vecLen) {
        // Perform computation
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* ptr, int vecLen) {
    std::srand(time({})); // use current time as seed for random generator
    for (int i = 0; i < vecLen; i++) {
        ptr[i] = rand() / (float)RAND_MAX;
    }
}

// Host addition of A and B and storing the result in C. All vectors are of length `len`.
void serialVecAdd(float* A, float* B, float* C, int len) {
    for (int i = 0; i < len; i++) {
        C[i] = A[i] + B[i];
    }
}

// Compared if the `left` and `right` vectors, both of length `len` are equal based on the `epsilon`
// precision
bool vectorApproximatelyEqual(float* left, float* right, int len, float epsilon=0.00001) {
    for (int i = 0; i < len; i++) {
        if (left[i] - right[i] > epsilon) {
            printf("Index %d mismatch: %f != %f\n", i, left[i], right[i]);
            return false; 
        }
    }
    return true;
}

// Utility macro to check for cuda errors
// TODO: This needs to become and error logging mechanism
// TODO: error reporting is async, check https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#asynchronous-execution-error-handling
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

// Number of threads that will execute the kernel in parallel is specified as part of the kernel
// launch. Different invocations of the same kernel may use different execution configurations such
// as different number of threads or thread blocks.

// There are 2 ways to launch kernels from CPU code:
// - triple chevron notation
// - cudaLaunchKernelEx

int main() {
    int vecLen = 1024;
    // Kernel invocation
    // 1 - TB, 256 - Threads
    // Each thread will execute the exact same kernel code.
    // Since all threads of a block reside on the same SM and must share resources, a TB may contain
    // up to 1024 threads. If resources allow, more than one TB can be schedules on an SM
    // simultaneously.

    // Kernel launches are async with respect to the host thread.
    // Kernel will be setup for execution on the GPU, but the hsot code will not wait for the kernel
    // to complete (or even start) executing on the GPU before proceeding.
    // Advanced async execution syncronization [https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#asynchronous-execution]

    // vecLen is storing the number of elements in any of the 3 vectors
    int threads = 256;
    // We chose the number of blocks, by rounding up a multiple of `threads` above vecLen.
    // Extra threads in a block that do no work does not incur a large overhead cost.
    int blocks = (vecLen + (threads - 1)) / threads;
    // Conveniently equal to the above is
    blocks = cuda::ceil_div(vecLen, threads);

    // Unified Memory is allocated (by the NVIDIA Driver) using `cudaMallocManaged` API or by
    // declaring a variable with the __managed__ specifier.
    // The NV Driver will make sure that the memory is accessible to the GPU or CPU whenever either
    // tries to access it.

    // Pointers to host memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    // Stores a CPU computed result for comparison
    float* comparisonResult = (float*)malloc(vecLen*sizeof(float));

    // Pointers to device memory vectors
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    /*
    // Use unified memory to allocate buffers - can be accessed from either the CPU or the GPU
    cudaMallocManaged(&A, vecLen*sizeof(float));
    cudaMallocManaged(&B, vecLen*sizeof(float));
    cudaMallocManaged(&C, vecLen*sizeof(float));
    */

    // Allocate memory on host CPU
    CUDA_CHECK(cudaMallocHost(&A, vecLen*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&B, vecLen*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&C, vecLen*sizeof(float)));

    // Initialize vectors on the host
    initArray(A, vecLen);
    initArray(B, vecLen);

    // Allocate memory on device GPU
    CUDA_CHECK(cudaMalloc(&devA, vecLen*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vecLen*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC, vecLen*sizeof(float)));

    // Copy host vecs to device
    cudaMemcpy(devA, A, vecLen*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, vecLen*sizeof(float), cudaMemcpyHostToDevice);
    // Init the result
    cudaMemset(devC, 0.0, vecLen*sizeof(float));

    // Launch the kernel. Unified memory will make sure A, B and C are accesible to the GPU
    threads = 256;
    // We chose the number of blocks, by rounding up a multiple of `threads` above vecLen.
    // Extra threads in a block that do no work does not incur a large overhead cost.
    blocks = (vecLen + (threads - 1)) / threads;
    // Launch kernel
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vecLen);
    // Kernel launches do not return error, so we much check if kernel launch parameters and
    // execution configuration passed successfuly. Since async operations in the CUDA runtime are
    // async, a success result does not guarantee a successful kernel launch or execution.
    CUDA_CHECK(cudaPeekAtLastError());

    // Wait for the device (kernel) to complete execution.
    CUDA_CHECK(cudaDeviceSynchronize());
    // Another basic mechanism for synchronization at the block level is the `__syncthreads()`
    // Syncs between TBs is supported in TB clusters through Cooperative Groups APIs
    // Best perf is achieved with sync within a thread block.

    // Copy the result from device back to device
    CUDA_CHECK(cudaMemcpy(C, devC, vecLen*sizeof(float), cudaMemcpyDeviceToHost));

    // Perform computation serially on CPU for comparison
    // This could be moved before waiting for the result.
    serialVecAdd(A, B, comparisonResult, vecLen);

    // Confirm that the CPU and GPU got the same answer
    if (vectorApproximatelyEqual(C, comparisonResult, vecLen)) {
        printf("Unified Memory: CPU and GPU answers match\n");
    } else {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Release the buffers
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    free(comparisonResult);
}