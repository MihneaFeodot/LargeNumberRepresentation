#pragma once
#include <cuda_runtime.h>
// Includem matematica modulară (calea depinde de unde ai pus fișierul, 
// dar cu -I./include în compilator, e suficient doar numele)
#include "montgomery.h"

// ==========================================
// KERNEL 1: BIT REVERSAL PERMUTATION
// ==========================================
// Acest kernel amestecă vectorul inițial pentru a pregăti algoritmul Cooley-Tukey.
// Transformă indexul 001 (1) în 100 (4), etc.

__global__ void bit_reverse_kernel(field_t* d_data, int n, int log_n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        unsigned int reversed = 0;
        unsigned int temp = tid;
        
        // Algoritmul de inversare a biților
        for (int i = 0; i < log_n; i++) {
            reversed = (reversed << 1) | (temp & 1);
            temp >>= 1;
        }
        
        // Facem swap doar dacă reversed > tid (ca să nu le mutăm de două ori)
        if (reversed > tid) {
            field_t val_a = d_data[tid];
            field_t val_b = d_data[reversed];
            d_data[tid] = val_b;
            d_data[reversed] = val_a;
        }
    }
}

// ==========================================
// KERNEL 2: NTT BUTTERFLY STAGE (OPTIMIZED)
// ==========================================
// Folosește Shared Memory și Padding pentru viteză maximă.
// Se apelează iterativ din main: m = 2, 4, 8 ... N

__global__ void ntt_stage_kernel(field_t* global_data, const field_t* twiddles, int m, int n) {
    // 1. Definim memoria partajată dinamică
    // Mărimea ei este setată din main.cu: (N + N/32) * sizeof(field_t)
    extern __shared__ field_t s_data[];

    int tid = threadIdx.x;
   // int global_tid = threadIdx.x + blockIdx.x * blockDim.x;

    // --- FAZA A: Încărcare din Global în Shared (Cu Padding) ---
    // Fiecare thread ia un element din VRAM și îl pune în Cache-ul L1 (Shared)
    // Folosim macro-ul PADDED_INDEX din montgomery.h pentru a evita conflictele de bancă
    
    // Presupunem că lansăm suficiente thread-uri să acopere N (sau buclăm)
    for (int i = tid; i < n; i += blockDim.x) {
        s_data[PADDED_INDEX(i)] = global_data[i];
    }

    __syncthreads(); // Așteptăm ca toți să termine încărcarea

    // --- FAZA B: Calculul Butterfly ---
    
    int half_m = m / 2;
    // Doar jumătate din thread-uri muncesc efectiv la un pas (fiecare procesează o pereche)
    int total_pairs = n / 2;

    // Calculăm pe ce pereche lucrează thread-ul curent (dacă e cazul)
    // Nota: Aici simplificăm puțin maparea. Într-o implementare multi-block complexă
    // maparea e mai dificilă, dar pentru un singur bloc mare (N=1024), asta merge perfect.
    
    for (int k_idx = tid; k_idx < total_pairs; k_idx += blockDim.x) {
        // Matematica de indici Cooley-Tukey
        int k = k_idx % half_m;         // Poziția în grup (pentru Twiddle)
        int group = k_idx / half_m;     // Care grup de fluturi
        int j = group * m + k;          // Indexul Superior (U)
        int i_idx = j + half_m;         // Indexul Inferior (V)

        // Citim din Shared Memory (folosind PADDING)
        field_t u = s_data[PADDED_INDEX(j)];
        field_t v = s_data[PADDED_INDEX(i_idx)];

        // Luăm Twiddle Factor-ul corespunzător
        // (Aici presupunem structura standard de twiddles)
        field_t w = twiddles[k * (n / m)]; 

        // === OPERAȚIILE MATEMATICE ===
        // Folosim funcțiile tale din montgomery.h
        
        // 1. Înmulțire Modulară: V * W
        field_t vw = montgomery_mul(v, w);

        // 2. Adunare și Scădere Modulară
        s_data[PADDED_INDEX(j)]     = add_mod(u, vw); // U + VW
        s_data[PADDED_INDEX(i_idx)] = sub_mod(u, vw); // U - VW
    }

    __syncthreads(); // Așteptăm ca toți să termine calculele

    // --- FAZA C: Salvare din Shared în Global ---
    // Scoatem datele din cache și le scriem înapoi în VRAM
    for (int i = tid; i < n; i += blockDim.x) {
        global_data[i] = s_data[PADDED_INDEX(i)];
    }
}