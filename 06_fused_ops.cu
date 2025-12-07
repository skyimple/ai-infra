#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// 1.1 前向传播CUDA Kernel
template<typename scalar_t>
__global__ void fused_add_layernorm_forward_kernel(
    scalar_t* output,
    const scalar_t* input1,
    const scalar_t* input2,
    const scalar_t* gamma,
    const scalar_t* beta,
    scalar_t* mean,
    scalar_t* rstd,
    int N, int C,  // N: batch_size * seq_len, C: hidden_size
    float eps) {
    
    // 每个线程块处理一个样本（N维度）
    int n = blockIdx.x;
    int c = threadIdx.x;
    
    // 共享内存存储每个warp的中间结果
    __shared__ float s_mem[32];
    
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    // 1. 计算add操作并累加统计量（使用Warp级归约优化）
    for (int i = c; i < C; i += blockDim.x) {
        float val = static_cast<float>(input1[n * C + i]) + 
                   static_cast<float>(input2[n * C + i]);
        sum += val;
        sq_sum += val * val;
        // 临时存储结果供后续使用
        s_mem[threadIdx.x] = val;
    }
    
    // Warp级归约求sum和sq_sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sq_sum += __shfl_down_sync(0xffffffff, sq_sum, offset);
    }
    
    // 第一个线程计算均值和方差
    if (threadIdx.x == 0) {
        float m = sum / C;
        float s = sqrtf(sq_sum / C - m * m + eps);
        mean[n] = m;
        rstd[n] = 1.0f / s;
    }
    __syncthreads();
    
    // 2. 应用LayerNorm: (x - mean) / rstd * gamma + beta
    float m = mean[n];
    float s = rstd[n];
    
    for (int i = c; i < C; i += blockDim.x) {
        float val = s_mem[threadIdx.x];  // 重用计算好的add结果
        float normalized = (val - m) * s;
        output[n * C + i] = static_cast<scalar_t>(
            normalized * static_cast<float>(gamma[i]) + 
            static_cast<float>(beta[i]));
    }
}

// 1.2 C++ Wrapper函数（CUDA版本）
void fused_add_layernorm_forward_cuda(
    float* output,
    const float* input1,
    const float* input2,
    const float* gamma,
    const float* beta,
    float* mean,
    float* rstd,
    int N, int C,
    float eps) {
    
    // 配置kernel launch参数
    dim3 blocks(N);  // 每个样本一个block
    dim3 threads(min(1024, ((C + 31) / 32) * 32));  // 对齐到Warp
    
    fused_add_layernorm_forward_kernel<float><<<blocks, threads>>>(
        output, input1, input2, gamma, beta, mean, rstd, N, C, eps);
    
    cudaDeviceSynchronize();
}

// 1.3 CPU回退版本（供参考对比）
void fused_add_layernorm_forward_cpu(
    float* output,
    const float* input1,
    const float* input2,
    const float* gamma,
    const float* beta,
    float* mean,
    float* rstd,
    int N, int C,
    float eps) {
    
    for (int n = 0; n < N; ++n) {
        // 计算mean
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += input1[n * C + c] + input2[n * C + c];
        }
        mean[n] = sum / C;
        
        // 计算std
        float sq_sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            float val = input1[n * C + c] + input2[n * C + c];
            float diff = val - mean[n];
            sq_sum += diff * diff;
        }
        rstd[n] = 1.0f / sqrt(sq_sum / C + eps);
        
        // 归一化
        for (int c = 0; c < C; ++c) {
            float val = input1[n * C + c] + input2[n * C + c];
            float normalized = (val - mean[n]) * rstd[n];
            output[n * C + c] = normalized * gamma[c] + beta[c];
        }
    }
}