#include <iostream>
#include <vector>
#include <chrono> 
#include "bigint.cuh" 
#include "ntt_kernel.cuh"
#include "verification.h"

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s la linia %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

int main() {
    // Testele au fost realizate cu num_limbs = 1024, 4096 si 8192
    size_t num_limbs = 8192; 
    
    size_t log_n = 0;
    size_t temp = num_limbs;
    while (temp > 1) { temp >>= 1; log_n++; }
    
    std::cout << "=== SPEEDUP TEST: GPU vs CPU ===\n";
    std::cout << "N=" << num_limbs << " (" << log_n << " stages)\n";

    // Setup matematic
    uint64_t R = (1ULL << 32) % P_MOCK;
    field_t R2 = ((unsigned __int128)R * R) % P_MOCK;
    field_t g = 5;
    field_t omega = pow_mod(g, (P_MOCK - 1) / num_limbs, P_MOCK);

    // Pregatire date
    BigInt* h_num = BigIntCUDA::allocate_host(num_limbs);
    for(size_t i = 0; i < num_limbs; i++) h_num->limbs[i] = montgomery_mul(i, R2);

    BigInt* d_num = BigIntCUDA::allocate_device(num_limbs);
    BigIntCUDA::copy_to_device(d_num, h_num);

    std::vector<field_t> h_twiddles(num_limbs);
    for(size_t i=0; i<num_limbs; i++) h_twiddles[i] = montgomery_mul(pow_mod(omega, i, P_MOCK), R2);

    field_t* d_twiddles;
    CHECK(cudaMalloc(&d_twiddles, num_limbs * sizeof(field_t)));
    CHECK(cudaMemcpy(d_twiddles, h_twiddles.data(), num_limbs * sizeof(field_t), cudaMemcpyHostToDevice));

    // Masurare timp pe GPU (doar kernel-urile)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    field_t* raw_ptr = (field_t*)d_num->limbs; 
    int threads = 256;
    int blocks = (num_limbs + threads - 1) / threads;
    size_t shared_mem_size = (num_limbs + (num_limbs >> 5)) * sizeof(field_t);

    cudaEventRecord(start);

    bit_reverse_kernel<<<blocks, threads>>>(raw_ptr, num_limbs, log_n);
    
    for (int m = 2; m <= num_limbs; m *= 2) {
        ntt_stage_kernel<<<1, threads, shared_mem_size>>>(raw_ptr, d_twiddles, m, num_limbs);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float gpu_time_sec = milliseconds / 1000.0f;

    std::cout << "\n[GPU] Timp executie: " << gpu_time_sec << " secunde\n";

    // Recuperare date pentru verificare
    BigIntCUDA::copy_to_host(h_num, d_num);

    // Masurare timp pe CPU
    std::cout << "[CPU] Rulare Reference NTT (Asteapta, e lent...)... \n";
    
    std::vector<uint32_t> input_data(num_limbs);
    for(size_t i=0; i<num_limbs; i++) input_data[i] = i; 

    Verifier v(128);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    std::vector<uint32_t> cpu_reference = v.compute_reference_ntt_naive(input_data, omega, P_MOCK);
    
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_diff = cpu_stop - cpu_start;
    double cpu_time_sec = cpu_diff.count();

    std::cout << "[CPU] Timp executie: " << cpu_time_sec << " secunde\n";

    // Calculare speedup
    double speedup = cpu_time_sec / gpu_time_sec;
    
    std::cout << "\n============================================\n";
    std::cout << "REZULTAT FINAL (SPEEDUP):\n";
    std::cout << "GPU Time: " << gpu_time_sec << " s\n";
    std::cout << "CPU Time: " << cpu_time_sec << " s\n";
    std::cout << ">>> SPEEDUP: " << speedup << "x <<<\n";
    std::cout << "============================================\n";

    // Verificare corectitudine
    if (montgomery_mul(h_num->limbs[0], 1) == cpu_reference[0]) {
        std::cout << "[Check] Rezultatele par consistente (Index 0 Match).\n";
    } else {
        std::cout << "[Check] EROARE: Rezultatele difera!\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK(cudaFree(d_twiddles));
    BigIntCUDA::free_bigint(d_num);
    BigIntCUDA::free_bigint(h_num);

    return 0;
}
