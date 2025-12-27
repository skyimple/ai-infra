#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <float.h>

template<int D>
__global__ void flash_attention_warp_kernel(
    const float* __restrict__ Q,  // [L, D] in HBM
    const float* __restrict__ K,  // [L, D] in HBM
    const float* __restrict__ V,  // [L, D] in HBM
    float* __restrict__ O,        // [L, D] in HBM
    int L,
    float scale                  // 1 / sqrt(D)
) {
    // --------------------------------------------------
    // 1. warp owns one query Q[i]
    // --------------------------------------------------
    int i   = blockIdx.x;     // query index
    int tid = threadIdx.x;    // lane id [0,31]

    static_assert(D % 32 == 0, "D must be divisible by 32");

    const float* q_ptr = Q + i * D;
    float*       o_ptr = O + i * D;

    // --------------------------------------------------
    // 2. Load Q into registers (HBM → registers)
    // --------------------------------------------------
    float q_reg[D / 32];
    #pragma unroll
    for (int d = 0; d < D / 32; ++d) {
        q_reg[d] = q_ptr[tid + d * 32];
    }

    // --------------------------------------------------
    // 3. Online softmax state (warp-shared, replicated)
    // --------------------------------------------------
    float m_i = -FLT_MAX;   // max so far
    float l_i = 0.0f;       // sum exp so far

    float o_reg[D / 32];
    #pragma unroll
    for (int d = 0; d < D / 32; ++d) {
        o_reg[d] = 0.0f;
    }

    // --------------------------------------------------
    // 4. Loop over all keys j
    // --------------------------------------------------
    for (int j = 0; j < L; ++j) {

        // ---- 4.1 compute s_ij = Q_i · K_j
        const float* k_ptr = K + j * D;

        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < D / 32; ++d) {
            dot += q_reg[d] * k_ptr[tid + d * 32];
        }

        // warp-level reduction (sum)
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, offset);
        }

        float s_ij = dot * scale;

        // ---- 4.2 warp-level max(s_ij)
        float m_ij = s_ij;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            m_ij = fmaxf(m_ij,
                         __shfl_xor_sync(0xffffffff, m_ij, offset));
        }

        // ---- 4.3 online softmax update
        float m_new = fmaxf(m_i, m_ij);
        float alpha = expf(m_i - m_new);
        float p_ij  = expf(s_ij - m_new);

        // warp-level sum exp
        float p_sum = p_ij;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            p_sum += __shfl_xor_sync(0xffffffff, p_sum, offset);
        }

        l_i = alpha * l_i + p_sum;

        // ---- 4.4 update output O_i
        const float* v_ptr = V + j * D;
        #pragma unroll
        for (int d = 0; d < D / 32; ++d) {
            o_reg[d] = alpha * o_reg[d]
                     + p_ij * v_ptr[tid + d * 32];
        }

        m_i = m_new;
    }

    // --------------------------------------------------
    // 5. Normalize and write back (register → HBM)
    // --------------------------------------------------
    float inv_l = 1.0f / l_i;
    #pragma unroll
    for (int d = 0; d < D / 32; ++d) {
        o_ptr[tid + d * 32] = o_reg[d] * inv_l;
    }
}
