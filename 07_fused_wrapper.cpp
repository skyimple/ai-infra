%%writefile fused_wrapper.cpp
#include <torch/extension.h>

// 声明 C++ 接口 (包含在 .cu 文件中)
int fused_add_tanh(const float* A, const float* B, float* C, int N);

// 编写 PyTorch-compatible 的 Python 调用接口
torch::Tensor fused_add_tanh_forward(
    torch::Tensor A,
    torch::Tensor B) {
    
    // 1. 检查 Tensor 格式
    AT_ASSERTM(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors!");
    AT_ASSERTM(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous!");
    AT_ASSERTM(A.dtype() == torch::kFloat32, "Inputs must be float32!");

    int N = A.numel();
    
    // 2. 准备输出 Tensor (注意：必须先创建好，再把指针给 CUDA)
    auto C = torch::empty_like(A);

    // 3. 获取底层数据指针 (float*)
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();
    
    // 4. 调用 C++ 封装的 CUDA 函数
    fused_add_tanh(A_ptr, B_ptr, C_ptr, N);

    return C;
}

// 5. 将 C++ 函数绑定到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_tanh", &fused_add_tanh_forward, "Fused A + tanh(B) forward (CUDA)");
}