#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <float.h>

template<int D>
__global__ void flash_attention_warp_kernel(
    const float* __restrict__ Q,  // [L, D]
    const float* __restrict__ K,  // [L, D]
    const float* __restrict__ V,  // [L, D]
    float* __restrict__ O,        // [L, D]
    int L,
    float scale                  // 1 / sqrt(D)
) {
    // --------------------------------------------------
    // 1. One warp computes one query Q[i]
    // --------------------------------------------------
    int i   = blockIdx.x;
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
    // 3. Online softmax state (replicated per lane)
    // --------------------------------------------------
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    float o_reg[D / 32] = {0.0f};

    // --------------------------------------------------
    // 4. Loop over keys (SERIAL over j)
    // --------------------------------------------------
    for (int j = 0; j < L; ++j) {

        // ---- 4.1 dot(Q_i, K_j)
        const float* k_ptr = K + j * D;

        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < D / 32; ++d) {
            dot += q_reg[d] * k_ptr[tid + d * 32];
        }

        // warp reduction → scalar s_ij
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, offset);
        }
        // before we handle V, we got the complete s_{ij}
        float s_ij = dot * scale;

        // ---- 4.2 online softmax (NO warp max!)
        float m_new = fmaxf(m_i, s_ij);
        float alpha = expf(m_i - m_new);
        float p_ij  = expf(s_ij - m_new);

        l_i = alpha * l_i + p_ij;

        // ---- 4.3 update output accumulator
        // 这里逐步更新的是S=Q*K^{T}的第i行的元素s_{ij}
        // 从而更新的是最终输入的O的第i行
        // 这一列的元素是S的第i行，同V的不同列的元素相乘产生的结果
        // 看似是不同列，但S的元素是逐个产生的
        // 最新的s_{ij}，同V中产生关联的元素，恰恰是V的第j行的不同元素
        // 这里每个thread的o_reg就是保存了O的第i行的元素

        // 这行控制了关注V的第j行
        const float* v_ptr = V + j * D; 
        // 每个thread负责更新的是V的第j行的
        // tid + d*32 序号的元素，也可以理解为tid + d*32列
        #pragma unroll
        for (int d = 0; d < D / 32; ++d) {
            o_reg[d] = alpha * o_reg[d]
                     + p_ij * v_ptr[tid + d * 32];
        }

        m_i = m_new;
    }

    // --------------------------------------------------
    // 5. Normalize and write back
    // --------------------------------------------------
    // 最后更新o_ptr
    // 每个thread更新的是tid开始，间隔32个位置，共D/32个元素
    float inv_l = 1.0f / l_i;
    #pragma unroll
    for (int d = 0; d < D / 32; ++d) {
        o_ptr[tid + d * 32] = o_reg[d] * inv_l;
    }
}
