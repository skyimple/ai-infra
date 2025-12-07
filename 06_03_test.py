import torch
import fused_layernorm
import time

def test_correctness():
    """测试正确性：对比PyTorch原生实现"""
    batch_size, seq_len, hidden_size = 2, 128, 768
    
    # 创建随机输入
    input1 = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    input2 = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    gamma = torch.randn(hidden_size, device='cuda')
    beta = torch.randn(hidden_size, device='cuda')
    
    # PyTorch原生实现
    torch_output = input1 + input2
    torch_output = torch.nn.functional.layer_norm(
        torch_output, [hidden_size], gamma, beta, 1e-5)
    
    # 自定义融合实现
    fused_output = fused_layernorm.forward(input1, input2, gamma, beta, 1e-5)
    
    # 计算最大误差
    diff = (fused_output - torch_output).abs().max()
    print(f"最大绝对误差: {diff.item():.6f}")
    
    # 验证是否在允许范围内
    assert diff < 1e-4, f"误差过大: {diff}"
    print("✅ 正确性测试通过")

def test_performance():
    """测试性能对比"""
    batch_size, seq_len, hidden_size = 8, 512, 1024
    
    input1 = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    input2 = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    gamma = torch.randn(hidden_size, device='cuda')
    beta = torch.randn(hidden_size, device='cuda')
    
    # 预热
    for _ in range(10):
        _ = fused_layernorm.forward(input1, input2, gamma, beta)
    
    # 测试自定义kernel
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        fused_output = fused_layernorm.forward(input1, input2, gamma, beta)
    torch.cuda.synchronize()
    fused_time = time.time() - start
    
    # 测试PyTorch原生
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        torch_output = input1 + input2
        torch_output = torch.nn.functional.layer_norm(
            torch_output, [hidden_size], gamma, beta)
    torch.cuda.synchronize()
    torch_time = time.time() - start
    
    print(f"自定义融合kernel耗时: {fused_time:.3f}s")
    print(f"PyTorch原生实现耗时: {torch_time:.3f}s")
    print(f"加速比: {torch_time/fused_time:.2f}x")

def test_memory():
    """测试内存使用"""
    batch_size, seq_len, hidden_size = 4, 256, 4096
    
    input1 = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    input2 = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    gamma = torch.randn(hidden_size, device='cuda')
    beta = torch.randn(hidden_size, device='cuda')
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # 记录峰值内存
    fused_output = fused_layernorm.forward(input1, input2, gamma, beta)
    torch.cuda.synchronize()
    fused_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    torch_output = input1 + input2
    torch_output = torch.nn.functional.layer_norm(
        torch_output, [hidden_size], gamma, beta)
    torch.cuda.synchronize()
    torch_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"自定义kernel峰值内存: {fused_memory:.2f} MB")
    print(f"PyTorch原生峰值内存: {torch_memory:.2f} MB")
    print(f"内存节省: {torch_memory - fused_memory:.2f} MB")

if __name__ == "__main__":
    print("=== 正确性测试 ===")
    test_correctness()
    
    print("\n=== 性能测试 ===")
    test_performance()
    
    print("\n=== 内存测试 ===")
    test_memory()