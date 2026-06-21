// Kernel definition
// Compile time kernel attribute thread block cluster size (2, 1, 1)
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float *outputa) {}

int main() {
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Grid dimmension is not affected by cluster launc and is still enumerated using number of
    // blocks
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}