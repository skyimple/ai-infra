#include <stdio.h>
#include <cuda_runtime.h>

// 这个函数运行在 GPU 上
__global__ void gpu_kernel(int* d_val) {
    // 这里的 *d_val 是在 GPU 显存里取值
    printf("GPU: Value at address %p is %d\n", d_val, *d_val);
    
    // 修改 GPU 显存里的值
    // 这里非常直觉，gpu_kernel核函数接受的参数是d_val
    // d_val是一个整数指针
    // 修改了d_val指向变量的数值，赋值为200
    *d_val = 200;
}

int main() {
    // h_a 为CPU上的一个变量
    int h_a = 100;      // CPU 上的变量 (Host)
    // 仅仅是一个wild pointer
    int *d_a;           // GPU 上的指针 (Device Pointer) - 此时它还是野指针

    printf("CPU: Original value is %d\n", h_a);

    // 1. 分配 GPU 显存
    // 难点：为什么是 (void**)&d_a ? 
    // 因为 cudaMalloc 需要修改 d_a 指针本身的值（让它指向显存地址），
    // 所以需要传入 d_a 的地址（即指针的指针）。
    cudaMalloc((void**)&d_a, sizeof(int));

    // 2. 将数据从 CPU 拷贝到 GPU
    // copy h_a 的值 到 d_a 指向的地址
    // 先设置d_a d_a的数值是一段GPU的显存地址
    cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);

    // 3. 启动 Kernel
    gpu_kernel<<<1, 1>>>(d_a);
    cudaDeviceSynchronize(); // 等待 GPU 跑完

    // 4. 将修改后的数据拷回 CPU
    // 注意：我们要把 d_a 指向的值，拷回 h_a 的地址 (&h_a)
    cudaMemcpy(&h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

    printf("CPU: Modified value is %d\n", h_a);

    // 5. 释放显存
    cudaFree(d_a);

    return 0;
}
