__global__ void fused_attention_v_kernel(float* Q, float* K, float* V, float* O, int seq_len, int d) {
    int row = blockIdx.x; // Each block handles one Query row
    int tid = threadIdx.x;

    // Local stats for Online Softmax
    float row_max = -1e20f;
    float row_sum = 0.0f;
    
    // Output accumulator (stored in registers)
    // For d=64, each thread might store a part of this vector
    extern __shared__ float sK[]; // Shared memory for K tiles
    extern __shared__ float sV[]; // Shared memory for V tiles
    float thread_O[64] = {0.0f};  // Adjust based on head_dim

    for (int tile_idx = 0; tile_idx < seq_len; tile_idx += TILE_SIZE) {
        // 1. Load K and V tiles into Shared Memory (Collaborative)
        load_tile_to_shared(K, sK, tile_idx, d);
        load_tile_to_shared(V, sV, tile_idx, d);
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j) {
            // Compute score = Q[row] * K[tile_j]
            float score = compute_dot_product(Q, sK, row, j, d) * scale;

            // --- ONLINE SOFTMAX + V FUSION ---
            float old_max = row_max;
            if (score > row_max) {
                row_max = score;
                float rescale = expf(old_max - row_max);
                row_sum = row_sum * rescale + expf(score - row_max);
                
                // Rescale the existing Output vector to the new max
                for(int i=0; i<d; ++i) thread_O[i] *= rescale;
            } else {
                row_sum += expf(score - row_max);
            }

            // Accumulate V into the running Output
            float p_unnormalized = expf(score - row_max);
            for(int i=0; i<d; ++i) {
                thread_O[i] += p_unnormalized * sV[j * d + i];
            }
        }
        __syncthreads();
    }

    // Final normalization: Divide by the total sum of exponentials
    for(int i=0; i<d; ++i) {
        O[row * d + i] = thread_O[i] / row_sum;
    }
}