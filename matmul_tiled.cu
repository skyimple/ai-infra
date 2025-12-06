%%writefile matmul_tiled.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024
#define TILE_WIDTH 16  // 块大小 16x16

__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
    // 1. 定义 Shared Memory
    // 这里的 static 大小是在编译时确定的
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    // 2. 计算线程索引
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int bx = blockIdx.x; 
    int by = blockIdx.y;

    // Row 和 Col 是当前线程负责计算的 C 矩阵元素的坐标
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    // 3. 核心循环：将 A 的行和 B 的列 分成多个 Tile 遍历
    // m 代表当前是第几个 Tile (phase)
    for (int m = 0; m < n / TILE_WIDTH; ++m) {

        // --- 阶段 A: 协作加载数据到 Shared Memory ---
        
        // TODO 1: 加载 ds_A
        // 这里的任务是：当前线程 (ty, tx) 需要帮团队搬运 A 矩阵中的哪一个元素？
        // A 的行是 row，列是 (m * TILE_WIDTH + tx)
        ds_A[ty][tx] = A[row * n + (m * TILE_WIDTH + tx)];

        // TODO 2: 加载 ds_B
        // B 的行是 (m * TILE_WIDTH + ty)，列是 col
        ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * n + col];

        // TODO 3: 确保所有线程都搬运完了，大家才能开始算
        // 这一行非常关键！
         __syncthreads();

        // --- 阶段 B: 在 Shared Memory 上进行计算 ---
        
        // TODO 4: 计算 Partial Sum
        // 现在大家不需要去 Global Memory 了，直接读 ds_A 和 ds_B
        for (int k = 0; k < TILE_WIDTH; ++k) {
             sum += ds_A[ty][k] * ds_B[k][tx];
        }

        // TODO 5: 确保所有线程都算完了，才能进入下一次循环去更新 Shared Memory
        // 否则有的线程跑得快，把 ds_A 覆盖了，别的线程还没算完当前这一轮
         __syncthreads();
    }

    // 4. 写回 Global Memory
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    // ... (Main 函数与之前基本一致，只需修改 Kernel 调用名为 matmul_tiled) ...
    // 为了节省篇幅，你可以直接复用 Milestone 3 的 main 函数逻辑
    // 记得修改 Kernel 调用： matmul_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    // 下面只提供 main 函数的差异部分，你需要补全完整的 main 才能运行
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    // ... 初始化 h_A, h_B ...
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    printf("Running Tiled MatMul...\n");
    // Warmup
    matmul_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matmul_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("Tiled Matrix Mul Time: %f ms\n", ms);
    
    // Verify
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("Verify C[0]: %f (Expected %f)\n", h_C[0], 2.0f * N);

    // Free...
    return 0;
}