__global__ void flash_attention_kernel(
    const half* Q, 
    const half* K, 
    const half* V, 
    half* O,
    int N, 
    int d, 
    float softmax_scale
) {
    // 1. Setup Shared Memory for Q, K, V block
    extern __shared__ half smem[];
    half* s_Q = smem;
    half* s_K = s_Q + Br * d;
    half* s_V = s_K + Bc * d;

    // 2. Initialize O, m, l in Registers
    float acc_o[Br][d] = {0.0f}; // Accumulator for Output
    float m[Br] = {-inf};        // Max stats
    float l[Br] = {0.0f};        // Sum stats

    // 3. Outer Loop: Iterate over Q blocks (Row blocks)
    // int tx = threadIdx.x; 
    // Load Q_block into s_Q
    
    // 4. Inner Loop: Iterate over K, V blocks (Col blocks)
    for (int j = 0; j < Tc; j++) {
        // a. Load K_j, V_j into s_K, s_V
        __syncthreads();

        // b. GEMM 1: S = Q * K^T
        // 注意：这里不用写回 Global Mem，直接在 Reg 里算
        
        // c. Online Softmax Update & Rescaling
        // m_prev = m;
        // m_new = max(m_prev, row_max(S));
        // l_new = ...
        // Rescale acc_o: acc_o *= exp(m_prev - m_new);
        
        // d. GEMM 2: acc_o += P * V
        // Update m, l
        
        __syncthreads();
    }

    // 5. Finalize and Write O to Global Memory
    // O[i] = acc_o / l
}