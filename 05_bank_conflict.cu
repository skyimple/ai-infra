%%writefile transpose_benchmark.cu
#include <stdio.h>
#include <cuda_runtime.h>

// 矩阵大小 2048 x 2048
// 选择 32 是为了最大化 Bank Conflict (WarpSize = 32)
#define N 2048
#define TILE_DIM 32

// ==========================================
// Kernel 1: Naive Copy (基准线)
// 读写都是合并的，理论带宽上限
// ==========================================
__global__ void copySharedMem(float *odata, const float *idata) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
        __syncthreads();
        odata[y * width + x] = tile[threadIdx.y][threadIdx.x];
    }
}

// ==========================================
// Kernel 2: Naive Transpose
// 致命弱点：写操作不合并 (Uncoalesced Write)
// 写入 Global Memory 时 stride = 2048，导致带宽极大浪费
// ==========================================
__global__ void transposeNaive(float *odata, const float *idata) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (x < N && y < N) {
        // 读: idata[y][x] (合并，快)
        // 写: odata[x][y] (不合并，极慢)
        odata[x * width + y] = idata[y * width + x];
    }
}

// ==========================================
// Kernel 3: Shared Memory Transpose (With Bank Conflict)
// 解决了 Global Write 不合并的问题，但引入了 Shared Memory 冲突
// ==========================================
__global__ void transposeShared(float *odata, const float *idata) {
    // 声明：没有 Padding
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // 1. 读取到 Shared Memory (合并读取，无冲突写入 SM)
    // tile[ty][tx] = idata[y][x]
    if (x < width && y < width) // 简化边界检查
        tile[threadIdx.y][threadIdx.x] = idata[y * width + x];

    __syncthreads();

    // 2. 计算转置后的目标索引
    // 我们希望写出时，依然利用 threadIdx.x 连续变化来实现 Global Memory 合并写
    // 所以我们改变了 x, y 的计算逻辑：blockIdx.y 变成了新的 x 块坐标
    x = blockIdx.y * TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < width && y < width) {
        // 3. 从 Shared Memory 读出并写入 Global Memory
        // 读: tile[tx][ty] 
        // 灾难现场：Warp 内 tx 变化，导致读取 tile[0][ty], tile[1][ty]...
        // 也就是访问同一列。由于 stride=32，全部命中同一个 Bank。
        odata[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ==========================================
// Kernel 4: Shared Memory Transpose + Padding (优化完全体)
// ==========================================
__global__ void transposeSharedPadding(float *odata, const float *idata) {
    // 唯一的改动：增加Padding列 [TILE_DIM + 1]
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (x < width && y < width)
        tile[threadIdx.y][threadIdx.x] = idata[y * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < width && y < width) {
        // 此时 tile[tx][ty] 的物理地址 stride 是 33
        // 33 % 32 = 1，Bank 错开，无冲突。
        odata[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// 辅助函数：运行并计时
void run_kernel(const char* name, void (*kernel)(float*, const float*), 
                float *d_out, float *d_in, dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // Warm up
    kernel<<<grid, block>>>(d_out, d_in);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i=0; i<100; i++) { // 跑 100 次取平均，放大差异
        kernel<<<grid, block>>>(d_out, d_in);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-25s: %f ms (Total for 100 runs)\n", name, ms);
    
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    size_t bytes = N * N * sizeof(float);
    float *h_in = (float*)malloc(bytes);
    float *h_check = (float*)malloc(bytes);
    float *d_in, *d_out;

    // 初始化
    for(int i=0; i<N*N; i++) h_in[i] = i;

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize(N/TILE_DIM, N/TILE_DIM);

    printf("Matrix Size: %d x %d, Tile: %d x %d\n", N, N, TILE_DIM, TILE_DIM);
    printf("Running benchmarks (100 iterations each)...\n");
    printf("--------------------------------------------------\n");

    // 1. Run Copy (Baseline)
    run_kernel("Naive Copy (Baseline)", copySharedMem, d_out, d_in, gridSize, blockSize);

    // 2. Run Naive Transpose
    run_kernel("Naive Transpose", transposeNaive, d_out, d_in, gridSize, blockSize);

    // 3. Run Shared (Conflict)
    run_kernel("Shared (Bank Conflict)", transposeShared, d_out, d_in, gridSize, blockSize);

    // 4. Run Shared (Padding)
    run_kernel("Shared (Padding)", transposeSharedPadding, d_out, d_in, gridSize, blockSize);

    // 简单验证结果正确性
    cudaMemcpy(h_check, d_out, bytes, cudaMemcpyDeviceToHost);
    bool correct = true;
    for(int i=0; i<N*N; i++) {
        int r = i / N; int c = i % N;
        if(h_check[c*N + r] != h_in[i]) {
            correct = false; 
            break; 
        }
    }
    printf("--------------------------------------------------\n");
    printf("Result Verification: %s\n", correct ? "PASSED" : "FAILED");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_check);
    return 0;
}