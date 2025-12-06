%%writefile matmul_naive.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// 矩阵尺寸假设为正方形 N x N
#define N 1024 

// CUDA Kernel: 计算 C 的一个元素 (row, col)
__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    // 1. 计算当前线程负责计算矩阵 C 的哪一行 (row) 和哪一列 (col)
    // 提示：y 对应 row, x 对应 col
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 边界检查：防止线程跑出矩阵范围
    if (row < n && col < n) {
        float sum = 0.0f;
        // 3. 计算点积：A 的第 row 行 与 B 的第 col 列
        for (int k = 0; k < n; k++) {
            // 核心挑战：将 2D 坐标 (row, k) 和 (k, col) 转换为 1D 索引
            // A[row][k] -> A[row * n + k]
            // B[k][col] -> B[k * n + col]
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    // Host memory
    // memory allocation on the host
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // 初始化矩阵
    // assign the value for h_A and h_B
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device memory
    // memory allocation on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy Host -> Device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 定义 2D 的 Block 和 Grid
    // 一个 Block 负责 16x16 个元素
    // the size of block, logical elements of the block is the thread
    dim3 blockSize(16, 16);
    // Grid 覆盖整个 N x N 矩阵
    // the element of the grid is block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (N + blockSize.y - 1) / blockSize.y);

    printf("Matrix Size: %d x %d\n", N, N);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    // 计时开始
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 启动 Kernel
    matmul_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // 计时结束
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time elapsed: %f ms\n", milliseconds);

    // 验证结果 (只抽查第一个元素)
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    // C[0][0] 应该是 1.0 * 2.0 * N = 2.0 * 1024 = 2048
    printf("C[0] = %f, Expected = %f\n", h_C[0], 2.0f * N);
    if (fabs(h_C[0] - 2.0f * N) < 1e-3) printf("Test PASSED\n");
    else printf("Test FAILED\n");

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}