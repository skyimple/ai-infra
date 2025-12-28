%%writefile flash_attn.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

// 静态常量定义
#define Br 64
#define Bc 64
#define d 64

// Kernel 声明与实现
__global__ void flash_attn_fwd_kernel(
    const half* __restrict__ Q, 
    const half* __restrict__ K, 
    const half* __restrict__ V, 
    float* __restrict__ O,
    int N, 
    float softmax_scale
) {
    int tx = threadIdx.x;
    int batch_id = blockIdx.x;
    int head_id  = blockIdx.y;
    int br_idx   = blockIdx.z;

    // 偏移量计算
    int qkv_offset = (batch_id * gridDim.y * N * d) + (head_id * N * d);
    const half* Q_block_ptr = Q + qkv_offset + (br_idx * Br * d);
    float* O_block_ptr = O + qkv_offset + (br_idx * Br * d);

    // Shared Memory
    extern __shared__ half s_mem[];
    half* s_Q = s_mem;
    half* s_K = s_Q + Br * d;
    half* s_V = s_K + Bc * d;

    // Accumulators
    float acc_o[d] = {0.0f};
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    // Load Q to Shared Memory
    if (tx < Br && (br_idx * Br + tx) < N) {
        for (int x = 0; x < d; x++) {
            s_Q[tx * d + x] = Q_block_ptr[tx * d + x];
        }
    }
    __syncthreads();

    // Inner Loop
    int Tr = (N + Bc - 1) / Bc;
    for (int j = 0; j < Tr; j++) {
        // Load K, V to Shared Memory
        if (tx < Bc && (j * Bc + tx) < N) {
            const half* k_base = K + qkv_offset + (j * Bc * d);
            const half* v_base = V + qkv_offset + (j * Bc * d);
            for (int x = 0; x < d; x++) {
                s_K[tx * d + x] = k_base[tx * d + x];
                s_V[tx * d + x] = v_base[tx * d + x];
            }
        }
        __syncthreads();

        // 核心计算逻辑 (你的实现)
        for (int jj = 0; jj < Bc; jj++) {
            if ((j * Bc + jj) >= N) break; // 边界保护

            float s_ij = 0.0f;
            for (int x = 0; x < d; x++) {
                s_ij += __half2float(s_Q[tx * d + x]) * __half2float(s_K[jj * d + x]);
            }
            s_ij *= softmax_scale;

            float m_new = fmaxf(m_i, s_ij);
            float alpha = expf(m_i - m_new);
            float p_ij  = expf(s_ij - m_new);
            
            for (int x = 0; x < d; x++) {
                acc_o[x] = acc_o[x] * alpha + p_ij * __half2float(s_V[jj * d + x]);
            }
            m_i = m_new;
            l_i = l_i * alpha + p_ij;
        }
        __syncthreads();
    }

    // Final Write back
    if (tx < Br && (br_idx * Br + tx) < N) {
        float inv_l = 1.0f / (l_i + 1e-6f);
        for (int x = 0; x < d; x++) {
            O_block_ptr[tx * d + x] = acc_o[x] * inv_l;
        }
    }
}

// C++ Wrapper (你的实现)
torch::Tensor flash_attn_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float softmax_scale) {
    // ... 保持你的检查代码 ...
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    auto O = torch::empty_like(Q, torch::kFloat32);
    dim3 block_dim(Br);
    dim3 grid_dim(B, H, (N + Br - 1) / Br);
    size_t smem_size = (Br + 2 * Bc) * d * sizeof(half);

    flash_attn_fwd_kernel<<<grid_dim, block_dim, smem_size>>>(
        (const half*)Q.data_ptr<at::Half>(),
        (const half*)K.data_ptr<at::Half>(),
        (const half*)V.data_ptr<at::Half>(),
        O.data_ptr<float>(),
        N, softmax_scale
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fwd", &flash_attn_fwd, "FlashAttention Forward (CUDA)");
}