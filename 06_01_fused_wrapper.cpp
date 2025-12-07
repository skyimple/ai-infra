#include <torch/extension.h>
#include <vector>

// 前向声明CUDA函数
void fused_add_layernorm_forward_cuda(
    float* output,
    const float* input1,
    const float* input2,
    const float* gamma,
    const float* beta,
    float* mean,
    float* rstd,
    int N, int C,
    float eps);

// 2.1 PyTorch Tensor包装函数
torch::Tensor fused_add_layernorm_forward(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float eps) {
    
    // 输入验证
    TORCH_CHECK(input1.dim() >= 2, "Input1 must have at least 2 dimensions");
    TORCH_CHECK(input1.sizes() == input2.sizes(), "Input sizes must match");
    TORCH_CHECK(gamma.size(0) == input1.size(-1), "Gamma size mismatch");
    TORCH_CHECK(beta.size(0) == input1.size(-1), "Beta size mismatch");
    
    // 获取输入维度
    auto sizes = input1.sizes().vec();
    int C = sizes.back();  // hidden_size
    sizes.pop_back();
    int N = 1;
    for (auto s : sizes) N *= s;  // batch_size * seq_len
    
    // 分配输出和中间结果
    auto output = torch::empty_like(input1);
    auto mean = torch::empty({N}, input1.options());
    auto rstd = torch::empty({N}, input1.options());
    
    // 获取数据指针
    float* output_ptr = output.data_ptr<float>();
    const float* input1_ptr = input1.data_ptr<float>();
    const float* input2_ptr = input2.data_ptr<float>();
    const float* gamma_ptr = gamma.data_ptr<float>();
    const float* beta_ptr = beta.data_ptr<float>();
    float* mean_ptr = mean.data_ptr<float>();
    float* rstd_ptr = rstd.data_ptr<float>();
    
    // 检查是否在CUDA设备上
    if (input1.is_cuda()) {
        fused_add_layernorm_forward_cuda(
            output_ptr, input1_ptr, input2_ptr, 
            gamma_ptr, beta_ptr,
            mean_ptr, rstd_ptr,
            N, C, eps);
    } else {
        // CPU版本（可以调用我们写的CPU实现）
        // 这里简化为调用PyTorch原生实现
        auto added = input1 + input2;
        output = torch::layer_norm(added, {C}, gamma, beta, eps);
    }
    
    // 返回输出（实际中可能还需要返回mean和rstd供反向传播用）
    return output;
}

// 2.2 定义PyTorch模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_add_layernorm_forward, 
          "Fused Add + LayerNorm forward",
          py::arg("input1"), 
          py::arg("input2"),
          py::arg("gamma"), 
          py::arg("beta"),
          py::arg("eps") = 1e-5);
}