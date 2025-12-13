#ifndef BIGINT_CUH
#define BIGINT_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

typedef uint32_t limb_t;
typedef uint64_t double_limb_t;

#define LIMB_BITS 32
#define LIMB_MAX 0xFFFFFFFF
#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct BigInt {
    limb_t* limbs;
    size_t num_limbs;
    bool is_negative;
    bool on_device;
    
    BigInt(size_t n_limbs = 0, bool negative = false);
    ~BigInt();
};

class BigIntCUDA {
public:
    static BigInt* allocate_device(size_t num_limbs);
    static BigInt* allocate_host(size_t num_limbs);
    static void free_bigint(BigInt* num);
    
    static void copy_to_device(BigInt* dst, const BigInt* src);
    static void copy_to_host(BigInt* dst, const BigInt* src);
    
    static BigInt* from_uint64(uint64_t value);
    static void print_bigint(const BigInt* num);
    
    static BigInt* add(const BigInt* a, const BigInt* b);
    static BigInt* subtract(const BigInt* a, const BigInt* b);
};

#endif
