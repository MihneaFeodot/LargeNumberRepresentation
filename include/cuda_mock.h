#pragma once

// Acest fișier permite compilarea codului CUDA pe CPU simplu (pentru verificare)
#ifndef __CUDACC__
    #include <cmath>
    #include <algorithm>
    #include <cstdint>

    // Definim macro-urile CUDA ca fiind goale pentru C++ standard
    #define __host__
    #define __device__
    #define __global__
    #define __forceinline__ inline
    #define __constant__ const
    
    // Simulăm tipuri vectoriale simple dacă e nevoie
    struct uint2 { uint32_t x, y; };
    struct uint4 { uint32_t x, y, z, w; };
#else
    // Dacă suntem în CUDA, includem librăria reală
    #include <cuda_runtime.h>
#endif