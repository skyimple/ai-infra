#include <cuda.h>
#include <cuda_runtime.h>

#define D 64        // head dimension
#define MAX_N 1024  // 最大 key 数（示例用）

// Q: [M, D]
// K: [N, D]
// V: [N, D]
// O: [M, D]

__global__ void fused_attention_kernel(
    const float* __restrict__ Q,   // HBM
    const float* __restrict__ K,   // HBM
    const float* __restrict__ V,   // HBM
    float* __restrict__ O,         // HBM
    int N
) {
    // --------------------------------------------------
    // block / thread 索引
    // --------------------------------------------------
    int q_idx = blockIdx.x;         // 当前 query
    int tid   = threadIdx.x;

    // --------------------------------------------------
    // shared memory
    // --------------------------------------------------
    __shared__ float smem_Q[D];         // Q 向量
    __shared__ float smem_scores[MAX_N]; // QK^T scores

    // --------------------------------------------------
    // 1. load Q → shared
    // --------------------------------------------------
    if (tid < D) {
        smem_Q[tid] = Q[q_idx * D + tid];   // HBM → shared
    }
    __syncthreads();

    // --------------------------------------------------
    // 2. compute QK^T
    // 每个线程算多个 key
    // --------------------------------------------------
    for (int k = tid; k < N; k += blockDim.x) {
        float acc = 0.0f;  // register

        #pragma unroll
        for (int d = 0; d < D; ++d) {
            acc += smem_Q[d] * K[k * D + d]; // shared × HBM → reg
        }

        smem_scores[k] = acc; // reg → shared
    }
    __syncthreads();

    // --------------------------------------------------
    // 3. softmax (naive two-pass)
    // --------------------------------------------------
    __shared__ float smem_max;
    __shared__ float smem_sum;

    if (tid == 0) {
        float max_val = -1e20f;
        for (int i = 0; i < N; ++i)
            max_val = fmaxf(max_val, smem_scores[i]);
        smem_max = max_val;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float e = __expf(smem_scores[i] - smem_max);
        smem_scores[i] = e;
        local_sum += e;
    }

    // block reduce
    atomicAdd(&smem_sum, local_sum);
    __syncthreads();

    // --------------------------------------------------
    // 4. weighted sum with V
    // O[q] = softmax(QK^T) @ V
    // --------------------------------------------------
    for (int d = tid; d < D; d += blockDim.x) {
        float out = 0.0f; // register
        for (int k = 0; k < N; ++k) {
            float w = smem_scores[k] / smem_sum;
            out += w * V[k * D + d];   // HBM
        }
        O[q_idx * D + d] = out; // reg → HBM
    }
}
