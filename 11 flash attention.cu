#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define D 64
#define BLOCK_K 128

__global__ void flash_attention_kernel(
    const float* Q,   // [L, D]  HBM
    const float* K,   // [L, D]  HBM
    const float* V,   // [L, D]  HBM
    float* O,         // [L, D]  HBM
    int L
) {
    // =====================================================
    // Indices
    // =====================================================
    int q_idx = blockIdx.x;   // one block per query
    int tid   = threadIdx.x;  // [0, BLOCK_K)

    // =====================================================
    // Shared memory
    // =====================================================
    __shared__ float K_tile[BLOCK_K][D];
    __shared__ float V_tile[BLOCK_K][D];

    // =====================================================
    // Registers (thread-private)
    // =====================================================
    float q_reg[D];           // Q_i
    float o_reg[D];           // output accumulator

    float m_i = -FLT_MAX;     // running max
    float l_i = 0.0f;         // running sum(exp)

    #pragma unroll
    for (int d = 0; d < D; ++d) {
        q_reg[d] = Q[q_idx * D + d];
        o_reg[d] = 0.0f;
    }

    // =====================================================
    // Loop over K/V blocks
    // =====================================================
    for (int k0 = 0; k0 < L; k0 += BLOCK_K) {

        int k_idx = k0 + tid;

        // -------------------------------------------------
        // Load K/V tiles (HBM → shared)
        // -------------------------------------------------
        if (k_idx < L) {
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                K_tile[tid][d] = K[k_idx * D + d];
                V_tile[tid][d] = V[k_idx * D + d];
            }
        }
        __syncthreads();

        // -------------------------------------------------
        // Compute QK^T for this tile
        // -------------------------------------------------
        if (k_idx < L) {
            float s_ij = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                s_ij += q_reg[d] * K_tile[tid][d];
            }
            s_ij /= sqrtf((float)D);

            // -------------------------------------------------
            // FlashAttention online softmax update
            // -------------------------------------------------
            float m_new = fmaxf(m_i, s_ij);
            float alpha = expf(m_i - m_new);
            float beta  = expf(s_ij - m_new);

            // update output
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                o_reg[d] = o_reg[d] * alpha + beta * V_tile[tid][d];
            }

            // update normalizer
            l_i = l_i * alpha + beta;
            m_i = m_new;
        }
        __syncthreads();
    }

    // =====================================================
    // Final normalization
    // =====================================================
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        O[q_idx * D + d] = o_reg[d] / l_i;
    }
}
