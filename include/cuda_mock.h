#pragma once

// Acest fisier permite compilarea codului CUDA pe CPU (pentru verificare)
#ifndef __CUDACC__
    #include <cmath>
    #include <algorithm>
    #include <cstdint>

    #define __host__
    #define __device__
    #define __global__
    #define __forceinline__ inline
    #define __constant__ const
    
    struct uint2 { uint32_t x, y; };
    struct uint4 { uint32_t x, y, z, w; };
#else
    // Daca suntem in CUDA, includem libraria reala
    #include <cuda_runtime.h>
#endif