#include <cuda_fp16.h>
#include <math.h>

#define D 64
#define TILE_K 128

__global__ void fused_self_attention(
    const half* Q,   // [L, D]  --- HBM
    const half* K,   // [L, D]  --- HBM
    const half* V,   // [L, D]  --- HBM
    half* O,         // [L, D]  --- HBM
    int L
) {
    // =====================================================
    // Indices
    // =====================================================
    int q_idx = blockIdx.x;     // query row
    int tid   = threadIdx.x;    // key index inside tile

    // =====================================================
    // Shared memory (block-wide)
    // =====================================================
    __shared__ half  Q_shared[D];            // shared
    __shared__ half  K_tile[TILE_K][D];      // shared
    __shared__ half  V_tile[TILE_K][D];      // shared
    __shared__ float score_tile[TILE_K];     // shared
    __shared__ float softmax_sum;             // shared

    // =====================================================
    // Registers (thread-private)
    // =====================================================
    float out_reg[D];   // registers

    #pragma unroll
    for (int d = 0; d < D; d++) {
        out_reg[d] = 0.0f;
    }

    // =====================================================
    // Load Q row: HBM → shared
    // =====================================================
    if (tid < D) {
        Q_shared[tid] = Q[q_idx * D + tid];
    }
    __syncthreads();

    // =====================================================
    // Loop over K/V tiles
    // =====================================================
    for (int k0 = 0; k0 < L; k0 += TILE_K) {

        int k_idx = k0 + tid;

        // -------------------------------------------------
        // Load K, V tiles: HBM → shared
        // -------------------------------------------------
        if (tid < TILE_K && k_idx < L) {
            #pragma unroll
            for (int d = 0; d < D; d++) {
                K_tile[tid][d] = K[k_idx * D + d];
                V_tile[tid][d] = V[k_idx * D + d];
            }
        }
        __syncthreads();

        // -------------------------------------------------
        // Compute Q · K^T
        // -------------------------------------------------
        if (tid < TILE_K && k_idx < L) {
            float score = 0.0f;  // register
            #pragma unroll
            for (int d = 0; d < D; d++) {
                score += __half2float(Q_shared[d]) *
                         __half2float(K_tile[tid][d]);
            }
            score /= sqrtf((float)D);
            score_tile[tid] = expf(score);   // shared
        }
        __syncthreads();

        // -------------------------------------------------
        // Reduce softmax denominator
        // -------------------------------------------------
        if (tid == 0) {
            float sum = 0.0f;
            int valid = min(TILE_K, L - k0);
            for (int t = 0; t < valid; t++) {
                sum += score_tile[t];
            }
            softmax_sum = sum;   // shared
        }
        __syncthreads();

        // -------------------------------------------------
        // Accumulate weighted V
        // -------------------------------------------------
        if (tid < TILE_K && k_idx < L) {
            float w = score_tile[tid] / softmax_sum;  // register
            #pragma unroll
            for (int d = 0; d < D; d++) {
                out_reg[d] += w * __half2float(V_tile[tid][d]);
            }
        }
        __syncthreads();
    }

    // =====================================================
    // Write output: registers → HBM
    // =====================================================
    if (tid < D) {
        O[q_idx * D + tid] = __float2half(out_reg[tid]);
    }
}
