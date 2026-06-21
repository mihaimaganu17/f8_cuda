import numpy as np
from numba import cuda
import cupy as cp

@cuda.jit
def my_kernel(input_array, output_array):
    # Within a kernel, each thread can access the parameters of the execution configuration as well
    # as the thread's index and thread block index within the grid.
    # cuda.threadIdx.[xyz] -> index of the thread within a thread block
    # cuda.blockDim.[xyz] -> size of the current TB the thread is in
    # cuda.blockIdx.[xyz] -> index of current TB the thread is in relative to the grid
    # cuda.gridDim.[xyz] -> size of the current grid 
    # unspecified x,y,z values default to 1 for sizes and 0 for indices 
    pass

@cuda.jit
def vec_add(A, B, C):
    """Element-wise (GPU thread based) addition of A + B and store the result in C. All elements are
    1-dim vectors"""
    # Get the index of the element the current thread must process
    # short hand for numba to compute index, where n is the number of dims
    # idx = cuda.grid(n)
    elemIdx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    C[elemIdx] = A[elemIdx] + B[elemIdx]


def main():
    # Launch the kernel - A TB may contain up to 1024 threads. More than one TB can be schedules on
    # an SM simultaneously.
    # my_kernel[num_thread_blocks, threads_per_block](in_array, out_array)

    # 2 or 3 dimensional CUDA kernel launches
    # my_kernel[(gridX, gridY), (blockX, blockY)](in_array, out_array)
    # print("Hello from cuda-py!")

    # Vectore size is not a power of 2 nor a multipel fo the block_size
    vector_size = 2**25 + 11
    device = cp.cuda.Device()

    # Device arrays
    # create 2 vecs of uniform random f32 values to be added
    a = cp.random.uniform(-1, 1, vector_size)
    b = cp.random.uniform(-1, 1, vector_size)
    # Stores the result - create the same shape as `a`
    c = cp.zeros_like(a)

    block_size = 256
    grid_size = int(np.ceil(vector_size/block_size))
    vec_add[grid_size, block_size](a, b, c)

    # Sync the CPU thread to ensure all the GPU work was completed
    device.synchronize()

    ## Copy all 3 array to the CPU
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)

    # perform the same operation on cpu
    host_result = a_np + b_np

    # test the 2 results (from GPU and CPU) equal
    np.testing.assert_array_almost_equal(c_np, host_result)

    # Print diagnostics and abort
    print("Test succeeded")


if __name__ == "__main__":
    # kernel = add_kernel.specialize(x, y, out)
    # print(kernel.inspect_ptx())
    main()
