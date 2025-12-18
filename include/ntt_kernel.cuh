#pragma once
#include <cuda_runtime.h>
#include "montgomery.h"

// Kernel 1: Bit reversal permutation  -> amesteca vectorul initial
//                                     -> transforma indexul 001 (1) in 100 (4) etc
__global__ void bit_reverse_kernel(field_t* d_data, int n, int log_n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        unsigned int reversed = 0;
        unsigned int temp = tid;
        
        for (int i = 0; i < log_n; i++) {
            reversed = (reversed << 1) | (temp & 1);
            temp >>= 1;
        }
        
        if (reversed > tid) {
            field_t val_a = d_data[tid];
            field_t val_b = d_data[reversed];
            d_data[tid] = val_b;
            d_data[reversed] = val_a;
        }
    }
}

// Kernel 2: NTT butterfly stage 
__global__ void ntt_stage_kernel(field_t* global_data, const field_t* twiddles, int m, int n) {
    extern __shared__ field_t s_data[];

    int tid = threadIdx.x;

    // Incarcare din global in shared
    for (int i = tid; i < n; i += blockDim.x) {
        s_data[PADDED_INDEX(i)] = global_data[i];
    }

    __syncthreads(); 

    // Calculul butterfly
    int half_m = m / 2;
    int total_pairs = n / 2;

    for (int k_idx = tid; k_idx < total_pairs; k_idx += blockDim.x) {
        int k = k_idx % half_m;         
        int group = k_idx / half_m;     
        int j = group * m + k;          
        int i_idx = j + half_m;         

        field_t u = s_data[PADDED_INDEX(j)];
        field_t v = s_data[PADDED_INDEX(i_idx)];

        field_t w = twiddles[k * (n / m)]; 

        field_t vw = montgomery_mul(v, w);

        s_data[PADDED_INDEX(j)]     = add_mod(u, vw); 
        s_data[PADDED_INDEX(i_idx)] = sub_mod(u, vw); 
    }

    __syncthreads(); 

    // Salvare din shared in global
    for (int i = tid; i < n; i += blockDim.x) {
        global_data[i] = s_data[PADDED_INDEX(i)];
    }
}
