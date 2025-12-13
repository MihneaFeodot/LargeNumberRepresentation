#include "bigint.cuh"

__device__ inline void add_with_carry(limb_t a, limb_t b, limb_t carry_in,
                                      limb_t* sum, limb_t* carry_out) {
    double_limb_t temp = (double_limb_t)a + (double_limb_t)b + (double_limb_t)carry_in;
    *sum = (limb_t)(temp & LIMB_MAX);
    *carry_out = (limb_t)(temp >> LIMB_BITS);
}

__device__ inline void sub_with_borrow(limb_t a, limb_t b, limb_t borrow_in,
                                       limb_t* diff, limb_t* borrow_out) {
    double_limb_t temp = (double_limb_t)a - (double_limb_t)b - (double_limb_t)borrow_in;
    *diff = (limb_t)(temp & LIMB_MAX);
    *borrow_out = (temp >> LIMB_BITS) ? 1 : 0;
}

__global__ void sequential_add_kernel(const limb_t* a, const limb_t* b,
                                      limb_t* result, size_t num_limbs) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        limb_t carry = 0;
        for (size_t i = 0; i < num_limbs; i++) {
            double_limb_t temp = (double_limb_t)a[i] + (double_limb_t)b[i] + (double_limb_t)carry;
            result[i] = (limb_t)(temp & LIMB_MAX);
            carry = (limb_t)(temp >> LIMB_BITS);
        }
    }
}

__global__ void sequential_subtract_kernel(const limb_t* a, const limb_t* b,
                                           limb_t* result, size_t num_limbs) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        limb_t borrow = 0;
        for (size_t i = 0; i < num_limbs; i++) {
            double_limb_t temp = (double_limb_t)a[i] - (double_limb_t)b[i] - (double_limb_t)borrow;
            result[i] = (limb_t)(temp & LIMB_MAX);
            borrow = (temp >> LIMB_BITS) ? 1 : 0;
        }
    }
}

__global__ void parallel_add_kernel(limb_t* a, limb_t* b, limb_t* res, uint8_t* carries, size_t n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        uint64_t sum = (uint64_t)a[idx] + b[idx];
        res[idx] = (limb_t)sum;
        carries[idx] = (sum >> 32) & 1; 
    }
}

__global__ void carry_propagation_kernel(limb_t* res, uint8_t* carries, size_t n) {
}

__global__ void parallel_subtract_kernel(limb_t* a, limb_t* b, limb_t* res, uint8_t* borrows, size_t n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int64_t diff = (int64_t)a[idx] - b[idx];
        res[idx] = (limb_t)diff;
        borrows[idx] = (diff < 0) ? 1 : 0;
    }
}

__global__ void borrow_propagation_kernel(limb_t* res, uint8_t* borrows, size_t n) {
}
