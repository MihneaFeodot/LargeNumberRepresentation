#pragma once
#include <cstdint>

#define PADDED_INDEX(i) ((i) + ((i) >> 5))
typedef uint32_t field_t;
static const field_t P_MOCK = 3221225473; 
static const field_t INV_P_MOCK = 3221225471; 

__host__ __device__ __forceinline__ field_t add_mod(field_t a, field_t b) {
    uint64_t res = (uint64_t)a + b;
    if (res >= P_MOCK) res -= P_MOCK;
    return (field_t)res;
}

__host__ __device__ __forceinline__ field_t sub_mod(field_t a, field_t b) {
    if (a >= b) return a - b;
    else return P_MOCK - (b - a);
}

__host__ __device__ __forceinline__ field_t montgomery_mul(field_t a, field_t b) {
    uint64_t product = (uint64_t)a * b;
    uint32_t m = (uint32_t)product * INV_P_MOCK;
    unsigned __int128 t_full = (unsigned __int128)product + (unsigned __int128)m * P_MOCK;
    uint64_t t = (uint64_t)(t_full >> 32);
    if (t >= P_MOCK) return (field_t)(t - P_MOCK);
    return (field_t)t;
}

__host__ __device__ inline field_t pow_mod(field_t base, uint32_t exp, field_t mod) {
    field_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) {
             unsigned __int128 temp = (unsigned __int128)res * base;
             res = (field_t)(temp % mod);
        }
        unsigned __int128 temp = (unsigned __int128)base * base;
        base = (field_t)(temp % mod);
        exp /= 2;
    }
    return res;
}
