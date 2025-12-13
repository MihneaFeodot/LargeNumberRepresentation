#include <iostream>
#include <vector>
#include "bigint.cuh" 
#include "ntt_kernel.cuh"
#include "verification.h"

// Macro pentru verificare erori CUDA
#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s la linia %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

int main() {
    size_t num_limbs = 1024; // Dimensiunea vectorului
    size_t log_n = 10;       // 2^10 = 1024

    std::cout << "=== INTEGRATION TEST FINAL: BigInt + NTT + Verification ===\n";

    // 1. GENERARE INPUT (0, 1, 2...)
    BigInt* h_num = BigIntCUDA::allocate_host(num_limbs);
    for(size_t i = 0; i < num_limbs; i++) h_num->limbs[i] = i;

    BigInt* d_num = BigIntCUDA::allocate_device(num_limbs);
    BigIntCUDA::copy_to_device(d_num, h_num);

    // 2. GENERARE TWIDDLE FACTORS (CORECT MATEMATIC)
    // Avem nevoie de un generator g=5 pentru P=3221225473
    // Omega = 5^((P-1)/N) mod P
    field_t g = 5;
    field_t omega = pow_mod(g, (P_MOCK - 1) / num_limbs, P_MOCK);
    
    std::cout << "[Setup] Generare Twiddles pentru N=" << num_limbs << ", Omega=" << omega << "...\n";

    std::vector<field_t> h_twiddles(num_limbs);
    for(size_t i=0; i<num_limbs; i++) {
        // Precalculăm puterile lui Omega: w^0, w^1, w^2...
        h_twiddles[i] = pow_mod(omega, i, P_MOCK);
    }

    field_t* d_twiddles;
    CHECK(cudaMalloc(&d_twiddles, num_limbs * sizeof(field_t)));
    CHECK(cudaMemcpy(d_twiddles, h_twiddles.data(), num_limbs * sizeof(field_t), cudaMemcpyHostToDevice));

    // 3. RULARE NTT PE GPU
    std::cout << "[NTT] Rulare kernel pe GPU...\n";
    field_t* raw_ptr = (field_t*)d_num->limbs; 

    // Bit Reversal
    int threads = 256;
    int blocks = (num_limbs + threads - 1) / threads;
    bit_reverse_kernel<<<blocks, threads>>>(raw_ptr, num_limbs, log_n);
    CHECK(cudaDeviceSynchronize());

    // Butterfly Stages
    // Aici se vede optimizarea ta cu PADDED_INDEX din montgomery.h
    size_t shared_mem_size = (num_limbs + (num_limbs >> 5)) * sizeof(field_t);
    
    for (int m = 2; m <= num_limbs; m *= 2) {
        // La fiecare pas, ajustam numarul de blocuri daca e nevoie
        // Pentru N=1024, un singur bloc de 512 threaduri e ideal, dar aici folosim grid general
        int ntt_blocks = (num_limbs / 2 + threads - 1) / threads;
        ntt_stage_kernel<<<ntt_blocks, threads, shared_mem_size>>>(raw_ptr, d_twiddles, m, num_limbs);
        CHECK(cudaDeviceSynchronize());
    }

    // 4. RECUPERARE REZULTAT
    BigIntCUDA::copy_to_host(h_num, d_num);
    
    // 5. VERIFICARE (GOLDEN MODEL)
    std::cout << "[Verification] Comparare cu CPU Reference (MPFR)...\n";
    
    std::vector<uint32_t> gpu_result(num_limbs);
    std::vector<uint32_t> input_data(num_limbs);
    for(size_t i=0; i<num_limbs; i++) {
        gpu_result[i] = h_num->limbs[i];
        input_data[i] = i; 
    }

    Verifier v(128);
    // Calculăm referința pe CPU folosind același Omega
    std::vector<uint32_t> cpu_reference = v.compute_reference_ntt_naive(input_data, omega, P_MOCK);

    bool match = true;
    for(size_t i=0; i<num_limbs; i++) {
        if(gpu_result[i] != cpu_reference[i]) {
            match = false;
            std::cout << "Mismatch la index " << i << ": GPU=" << gpu_result[i] << " CPU=" << cpu_reference[i] << "\n";
            // Afișăm doar primele erori
            if (i > 5) break; 
        }
    }

    if (match) std::cout << "\n[SUCCESS] Rezultatele GPU sunt IDENTICE cu cele CPU!\n";
    else std::cout << "\n[FAIL] Rezultatele diferă.\n";

    CHECK(cudaFree(d_twiddles));
    BigIntCUDA::free_bigint(d_num);
    BigIntCUDA::free_bigint(h_num);

    return 0;
}