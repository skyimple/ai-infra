#include <cuda_fp16.h>
#include <math.h>

#define TILE_K 128        // tile size along sequence length
#define D 64              // embedding dimension (assumed fixed for simplicity)

__global__ void fused_self_attention(
    const half* Q,   // [L, D]  --- HBM (global memory)
    const half* K,   // [L, D]  --- HBM
    const half* V,   // [L, D]  --- HBM
    half* O,         // [L, D]  --- HBM (output)
    int L
) {
    // ============================================================
    // Block / thread indexing
    // ============================================================

    int q_idx = blockIdx.x;      // which query row we compute
    int tid   = threadIdx.x;     // thread id inside block

    // ============================================================
    // Shared memory (on-chip SRAM, shared by block)
    // ============================================================

    __shared__ half K_tile[TILE_K][D];   // shared memory
    __shared__ half V_tile[TILE_K][D];   // shared memory
    __shared__ float score_tile[TILE_K]; // shared memory (QK scores)

    // ============================================================
    // Registers (private to each thread)
    // ============================================================

    float q_reg[D];        // registers: one query vector
    float out_reg[D];      // registers: output accumulator
    float sum_exp = 0.0f;  // registers: softmax denominator

    // initialize output accumulator
    #pragma unroll
    for (int d = 0; d < D; d++) {
        out_reg[d] = 0.0f;
    }

    // ============================================================
    // Load Q[q_idx, :] into registers
    // ============================================================
    // Q is in HBM → registers
    if (tid < D) {
        q_reg[tid] = __half2float(Q[q_idx * D + tid]);
    }
    __syncthreads();

    // ============================================================
    // Loop over K/V tiles
    // ============================================================

    for (int k0 = 0; k0 < L; k0 += TILE_K) {

        int k_idx = k0 + tid;

        // --------------------------------------------------------
        // Load K and V tiles from HBM → shared memory
        // --------------------------------------------------------
        if (k_idx < L) {
            #pragma unroll
            for (int d = 0; d < D; d++) {
                K_tile[tid][d] = K[k_idx * D + d];  // HBM → shared
                V_tile[tid][d] = V[k_idx * D + d];  // HBM → shared
            }
        }
        __syncthreads();

        // --------------------------------------------------------
        // Compute Q · K^T  (dot product)
        // Each thread computes score for one key
        // --------------------------------------------------------
        if (k_idx < L) {
            float score = 0.0f;   // register
            #pragma unroll
            for (int d = 0; d < D; d++) {
                score += q_reg[d] * __half2float(K_tile[tid][d]);
            }
            score /= sqrtf((float)D);
            score_tile[tid] = score;   // register → shared
        }
        __syncthreads();

        // --------------------------------------------------------
        // Softmax (naïve, block-wide)
        // --------------------------------------------------------
        if (k_idx < L) {
            float exp_score = expf(score_tile[tid]);  // register
            score_tile[tid] = exp_score;               // shared
        }
        __syncthreads();

        // reduce sum of exp
        if (tid == 0) {
            float tmp = 0.0f;  // register
            for (int t = 0; t < TILE_K && (k0 + t) < L; t++) {
                tmp += score_tile[t];
            }
            sum_exp += tmp;   // register
        }
        __syncthreads();

        // --------------------------------------------------------
        // Accumulate weighted V
        // --------------------------------------------------------
        if (k_idx < L) {
            float weight = score_tile[tid];  // shared → register
            #pragma unroll
            for (int d = 0; d < D; d++) {
                out_reg[d] += weight * __half2float(V_tile[tid][d]);
            }
        }
        __syncthreads();
    }

    // ============================================================
    // Normalize by softmax denominator and write output
    // ============================================================
    // registers → HBM
    if (tid < D) {
        O[q_idx * D + tid] = __float2half(out_reg[tid] / sum_exp);
    }
}
