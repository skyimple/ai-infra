#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA Kernel Function
__global__ void vector_add(const float *A, const float *B, float *C, int N) {
    // 1. 计算当前线程的全局唯一索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 2. 计算整个 Grid 一次能处理多少个数据 (步长)
    int stride = gridDim.x * blockDim.x;

    // 3. Grid-Stride Loop 模式
    // 如果 idx < N，就计算；然后 idx 跳过 stride 长度，处理下一个数据
    // 这样即使 N > 线程总数，也能循环处理完所有数据
    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000; // 100万个元素
    size_t bytes = N * sizeof(float);

    // Host memory pointers
    float *h_A, *h_B, *h_C;
    // Device memory pointers
    float *d_A, *d_B, *d_C;

    // Allocate Host Memory (用 new 或者 malloc)
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);

    // Initialize data
    for(int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate Device Memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy Host -> Device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);


    // Launch Kernel
    // Block 大小通常设为 128, 256 或 512
    int blockSize = 256;
    // Grid 大小：我们要处理 N 个数据，需要多少个 Block？
    // 公式：(N + blockSize - 1) / blockSize 是为了向上取整
    int numBlocks = (N + blockSize - 1) / blockSize;

    printf("Launching kernel with %d blocks and %d threads per block\n", numBlocks, blockSize);
    
    vector_add<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Sync is required to ensure GPU is finished before copying back
    cudaDeviceSynchronize();

    // Copy Device -> Host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify correct
    double maxError = 0.0;
    for(int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(h_C[i] - 3.0f));
    }
    printf("Max Error: %f\n", maxError);
    if(maxError < 1e-5) printf("Test PASSED\n");
    else printf("Test FAILED\n");

    // Free Memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}