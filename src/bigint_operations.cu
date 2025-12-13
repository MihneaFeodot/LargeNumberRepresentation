#include "bigint.cuh"
#include <algorithm>

// =========================================================
// FIX: FORWARD DECLARATIONS (Spunem compilatorului că există)
// =========================================================
__global__ void parallel_add_kernel(limb_t* a, limb_t* b, limb_t* res, uint8_t* carries, size_t n);
__global__ void carry_propagation_kernel(limb_t* res, uint8_t* carries, size_t n);

__global__ void parallel_subtract_kernel(limb_t* a, limb_t* b, limb_t* res, uint8_t* borrows, size_t n);
__global__ void borrow_propagation_kernel(limb_t* res, uint8_t* borrows, size_t n);

// =========================================================
// IMPLEMENTAREA FUNCȚIILOR HOST
// =========================================================

BigInt* BigIntCUDA::add(const BigInt* a, const BigInt* b) {
    size_t max_limbs = std::max(a->num_limbs, b->num_limbs);
    size_t result_limbs = max_limbs + 1;
    
    BigInt* d_a = allocate_device(max_limbs);
    BigInt* d_b = allocate_device(max_limbs);
    BigInt* d_result = allocate_device(result_limbs);
    
    uint8_t* d_carries;
    CUDA_CHECK(cudaMalloc(&d_carries, result_limbs * sizeof(uint8_t)));
    
    if (!a->on_device) {
        CUDA_CHECK(cudaMemcpy(d_a->limbs, a->limbs, a->num_limbs * sizeof(limb_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_a->limbs, a->limbs, a->num_limbs * sizeof(limb_t), cudaMemcpyDeviceToDevice));
    }
    
    if (!b->on_device) {
        CUDA_CHECK(cudaMemcpy(d_b->limbs, b->limbs, b->num_limbs * sizeof(limb_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_b->limbs, b->limbs, b->num_limbs * sizeof(limb_t), cudaMemcpyDeviceToDevice));
    }
    
    int num_blocks = (max_limbs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Acum compilatorul stie ce e 'parallel_add_kernel' datorita declaratiei de sus
    parallel_add_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_a->limbs, d_b->limbs, d_result->limbs, d_carries, max_limbs);
    carry_propagation_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_result->limbs, d_carries, result_limbs);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaFree(d_carries);
    free_bigint(d_a);
    free_bigint(d_b);
    
    return d_result;
}

BigInt* BigIntCUDA::subtract(const BigInt* a, const BigInt* b) {
    size_t max_limbs = std::max(a->num_limbs, b->num_limbs);
    
    BigInt* d_a = allocate_device(max_limbs);
    BigInt* d_b = allocate_device(max_limbs);
    BigInt* d_result = allocate_device(max_limbs);
    
    uint8_t* d_borrows;
    CUDA_CHECK(cudaMalloc(&d_borrows, max_limbs * sizeof(uint8_t)));
    
    if (!a->on_device) {
        CUDA_CHECK(cudaMemcpy(d_a->limbs, a->limbs, a->num_limbs * sizeof(limb_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_a->limbs, a->limbs, a->num_limbs * sizeof(limb_t), cudaMemcpyDeviceToDevice));
    }
    
    if (!b->on_device) {
        CUDA_CHECK(cudaMemcpy(d_b->limbs, b->limbs, b->num_limbs * sizeof(limb_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_b->limbs, b->limbs, b->num_limbs * sizeof(limb_t), cudaMemcpyDeviceToDevice));
    }
    
    int num_blocks = (max_limbs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    parallel_subtract_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_a->limbs, d_b->limbs, d_result->limbs, d_borrows, max_limbs);
    borrow_propagation_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_result->limbs, d_borrows, max_limbs);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaFree(d_borrows);
    free_bigint(d_a);
    free_bigint(d_b);
    
    return d_result;
}