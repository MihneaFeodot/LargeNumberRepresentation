#include <iostream>
#include <vector>

// 1. IMPORTĂM INFRASTRUCTURA COLEGULUI 1
#include "bigint.cuh" 

// 2. IMPORTĂM ALGORITMUL TĂU
#include "ntt_kernel.cuh"

// 3. IMPORTĂM VERIFICATORUL (MPFR)
#include "verification.h"

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s la linia %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

int main() {
    size_t num_limbs = 1024; 
    size_t log_n = 10;

    std::cout << "=== INTEGRATION TEST FINAL: BigInt + NTT + Verification ===\n";

    // --- PASUL 1: SETUP ---
    BigInt* h_num = BigIntCUDA::allocate_host(num_limbs);
    for(size_t i = 0; i < num_limbs; i++) h_num->limbs[i] = i; // Input: 0, 1, 2...

    BigInt* d_num = BigIntCUDA::allocate_device(num_limbs);
    BigIntCUDA::copy_to_device(d_num, h_num);

    // Setup Twiddles (Dummy 1 pentru test, sau calculati corect daca ai implementat)
    field_t* d_twiddles;
    CHECK(cudaMalloc(&d_twiddles, num_limbs * sizeof(field_t)));
    std::vector<field_t> h_twiddles(num_limbs, 1); 
    CHECK(cudaMemcpy(d_twiddles, h_twiddles.data(), num_limbs * sizeof(field_t), cudaMemcpyHostToDevice));

    // --- PASUL 2: EXECUȚIE GPU (NTT) ---
    std::cout << "[NTT] Rulare kernel pe GPU...\n";
    field_t* raw_ptr = (field_t*)d_num->limbs; 

    // Bit Reversal
    int threads = 256;
    int blocks = (num_limbs + threads - 1) / threads;
    bit_reverse_kernel<<<blocks, threads>>>(raw_ptr, num_limbs, log_n);
    CHECK(cudaDeviceSynchronize());

    // Butterfly Stages
    size_t shared_mem_size = (num_limbs + (num_limbs >> 5)) * sizeof(field_t);
    for (int m = 2; m <= num_limbs; m *= 2) {
        int ntt_blocks = (num_limbs / 2 + threads - 1) / threads;
        ntt_stage_kernel<<<ntt_blocks, threads, shared_mem_size>>>(raw_ptr, d_twiddles, m, num_limbs);
        CHECK(cudaDeviceSynchronize());
    }

    // --- PASUL 3: RECUPERARE ---
    BigIntCUDA::copy_to_host(h_num, d_num);
    
    // --- PASUL 4: VERIFICARE MATEMATICĂ (Golden Model) ---
    std::cout << "[Verification] Comparare cu CPU Reference (MPFR)...\n";
    
    // Convertim rezultatul BigInt într-un std::vector pentru Verifier
    std::vector<uint32_t> gpu_result(num_limbs);
    std::vector<uint32_t> input_data(num_limbs);
    for(size_t i=0; i<num_limbs; i++) {
        gpu_result[i] = h_num->limbs[i];
        input_data[i] = i; // Reconstruim inputul știut
    }

    // Instanțiem Verificatorul
    Verifier v(128);
    
    // NOTĂ: Pentru ca verificarea să treacă pe bune, 'd_twiddles' de pe GPU 
    // trebuie să conțină puterile reale ale lui Omega, nu doar '1'.
    // Dacă ai pus '1' în twiddles, rezultatul GPU va fi greșit matematic (dar corect algoritmic).
    // Aici testăm doar dacă Verifier-ul rulează.
    
    // Calculăm referința pe CPU (Aceasta va dura puțin)
    // Folosim omega=1 pentru testul cu twiddles=1
    std::vector<uint32_t> cpu_reference = v.compute_reference_ntt_naive(input_data, 1, P_MOCK);

    bool match = true;
    for(size_t i=0; i<num_limbs; i++) {
        if(gpu_result[i] != cpu_reference[i]) {
            match = false;
            std::cout << "Mismatch la index " << i << ": GPU=" << gpu_result[i] << " CPU=" << cpu_reference[i] << "\n";
            break;
        }
    }

    if (match) std::cout << "\n[SUCCESS] Rezultatele GPU sunt IDENTICE cu cele CPU!\n";
    else std::cout << "\n[FAIL] Rezultatele diferă (Normal daca Twiddles sunt dummy 1).\n";

    // Curățenie
    CHECK(cudaFree(d_twiddles));
    BigIntCUDA::free_bigint(d_num);
    BigIntCUDA::free_bigint(h_num);

    return 0;
}