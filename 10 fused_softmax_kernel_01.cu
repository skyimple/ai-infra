%%writefile fused_softmax_kernel.cu
#include <cuda_runtime.h>
#include <math.h>

// N: 矩阵行数 (如 Batch * Head 数量)
// T: 矩阵列数 (Sequence Length)
// S: 输入 Attention Score [N, T]
// P: 输出 Probability [N, T]

__global__ void fused_row_softmax(const float* S, float* P, int N, int T) {
    // 每一个 Block 负责 Softmax 的一行 (Row)
    int row_idx = blockIdx.x; // Block ID 就是行索引

    if (row_idx >= N) return;

    const float* s_row = S + row_idx * T;
    float* p_row = P + row_idx * T;

    // --- 步骤 1: 第一次 Pass (计算 Max) ---
    float max_val = -1e20f; // 初始极小值
    for (int col = threadIdx.x; col < T; col += blockDim.x) {
        // 确保每个线程都参与到最大值的计算
        // TODO: 1. 找到该行 (s_row) 的最大值 max_val
        // max_val = max(max_val, s_row[col]); 
    }
    // TODO: 2. 在 Block 内使用 __syncthreads() 和原子操作/Warp Reduce，将 max_val 归约到整个 Block。
    // ...

    // --- 步骤 2: 第二次 Pass (计算 Sum 和 Final P) ---
    __syncthreads(); // 确保所有线程都拿到最终的 max_val

    float sum_exp = 0.0f;
    for (int col = threadIdx.x; col < T; col += blockDim.x) {
        // TODO: 3. 计算 sum_exp = sum(exp(s_row[col] - max_val))
        // ...
    }
    // TODO: 4. 在 Block 内归约 sum_exp 到最终的 Softmax 分母 L = sum_exp
    // ...

    // --- 步骤 步骤 3: 最终计算 ---
    __syncthreads(); // 确保所有线程都拿到最终的 L

    float inv_L = 1.0f / L; // 最终分母的倒数
    for (int col = threadIdx.x; col < T; col += blockDim.x) {
        // TODO: 5. 计算 P[i] = exp(s_row[col] - max_val) * inv_L
        // ...
    }
}