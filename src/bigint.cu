#include "bigint.cuh"
#include <cstdio>
#include <cstring>

BigInt::BigInt(size_t n_limbs, bool negative)
    : limbs(nullptr), num_limbs(n_limbs), is_negative(negative), on_device(false) {
    if (n_limbs > 0) {
        limbs = new limb_t[n_limbs];
        memset(limbs, 0, n_limbs * sizeof(limb_t));
    }
}

BigInt::~BigInt() {
    if (limbs != nullptr) {
        if (on_device) {
            cudaFree(limbs);
        } else {
            delete[] limbs;
        }
    }
}

BigInt* BigIntCUDA::allocate_device(size_t num_limbs) {
    BigInt* num = new BigInt(0);
    num->num_limbs = num_limbs;
    num->on_device = true;
    CUDA_CHECK(cudaMalloc(&num->limbs, num_limbs * sizeof(limb_t)));
    CUDA_CHECK(cudaMemset(num->limbs, 0, num_limbs * sizeof(limb_t)));
    return num;
}

BigInt* BigIntCUDA::allocate_host(size_t num_limbs) {
    return new BigInt(num_limbs);
}

void BigIntCUDA::free_bigint(BigInt* num) {
    if (num != nullptr) delete num;
}

void BigIntCUDA::copy_to_device(BigInt* dst, const BigInt* src) {
    CUDA_CHECK(cudaMemcpy(dst->limbs, src->limbs, 
                          src->num_limbs * sizeof(limb_t), cudaMemcpyHostToDevice));
    dst->is_negative = src->is_negative;
}

void BigIntCUDA::copy_to_host(BigInt* dst, const BigInt* src) {
    CUDA_CHECK(cudaMemcpy(dst->limbs, src->limbs,
                          src->num_limbs * sizeof(limb_t), cudaMemcpyDeviceToHost));
    dst->is_negative = src->is_negative;
}

BigInt* BigIntCUDA::from_uint64(uint64_t value) {
    BigInt* num = allocate_host(2);
    num->limbs[0] = (limb_t)(value & 0xFFFFFFFF);
    num->limbs[1] = (limb_t)(value >> 32);
    return num;
}

void BigIntCUDA::print_bigint(const BigInt* num) {
    if (num == nullptr) { printf("NULL\n"); return; }
    
    BigInt* h = const_cast<BigInt*>(num);
    if (num->on_device) {
        h = allocate_host(num->num_limbs);
        copy_to_host(h, num);
    }
    
    printf("0x");
    bool started = false;
    for (int i = h->num_limbs - 1; i >= 0; i--) {
        if (h->limbs[i] != 0 || started) {
            printf(started ? "%08x" : "%x", h->limbs[i]);
            started = true;
        }
    }
    if (!started) printf("0");
    printf("\n");
    
    if (num->on_device) free_bigint(h);
}
