%%writefile fused_op.cu
#include <cuda_runtime.h>
#include <math.h>

// 融合算子：Z[i] = A[i] + tanh(B[i])
__global__ void fused_add_tanh_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; i < N; i += stride) {
        // 在 Registers 中完成所有计算，无需中间结果写回 Global Memory
        C[i] = A[i] + tanhf(B[i]); // tanhf 是 CUDA C++ 中的 float 版本 tanh
    }
}

// C++ Wrapper: 供 Python 调用的接口
int fused_add_tanh(const float* A, const float* B, float* C, int N) {
    // 线程配置
    // 这里的blocksize 相当于一个block有多少个线程
    int blockSize = 256;
    // 这里的numBlocks相当于，gird中有多少个block
    int numBlocks = (N + blockSize - 1) / blockSize;

    fused_add_tanh_kernel<<<numBlocks, blockSize>>>(A, B, C, N);
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
        return 1; // 1 代表错误
    }

    return 0; // 0 代表成功
}