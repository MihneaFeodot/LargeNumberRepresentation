#pragma once
#include <cstdint>

#ifndef __CUDACC__
    #include "cuda_mock.h"
#endif

// Rezolvarea conflictelor de bancă pentru NTT
// Adaugă un "padding" (un spațiu gol) la fiecare 32 de elemente
// Astfel, indecșii care ar cădea pe aceeași bancă sunt decalați.
// Exemplu: În loc să aloce shared_mem[1024], el va aloca shared_mem[1024 + 32]
#define PADDED_INDEX(i) ((i) + ((i) >> 5))

typedef uint32_t field_t;

// === CONSTANTE CORECTATE ===
// P = 3221225473 (0xC0000001)
static const field_t P_MOCK = 3221225473; 

// INV_P calculat corect: -P^(-1) mod 2^32
// Verificare: 3221225473 * 3221225471 = -1 (mod 2^32)
static const field_t INV_P_MOCK = 3221225471; 

// === OPERAȚII MODULARE DE BAZĂ (ADD/SUB) ===

// Adunare Modulară: (a + b) % P
__host__ __device__ __forceinline__ field_t add_mod(field_t a, field_t b) {
    // Folosim uint64_t pentru calcul intermediar deoarece a + b poate depăși 2^32
    // (P este mare, deci 2*P > UINT32_MAX)
    uint64_t res = (uint64_t)a + b;
    if (res >= P_MOCK) {
        res -= P_MOCK;
    }
    return (field_t)res;
}

// Scădere Modulară: (a - b) % P
__host__ __device__ __forceinline__ field_t sub_mod(field_t a, field_t b) {
    if (a >= b) {
        return a - b;
    } else {
        // Dacă a < b, rezultatul ar fi negativ.
        // Calculăm matematic (a - b) + P.
        // Pentru a evita overflow la (a + P), facem P - (b - a).
        return P_MOCK - (b - a);
    }
}

// === ÎNMULȚIRE MONTGOMERY ===

__host__ __device__ __forceinline__ field_t montgomery_mul(field_t a, field_t b) {
    uint64_t product = (uint64_t)a * b;
    
    // Calculăm m = (product * INV_P) mod 2^32
    uint32_t m = (uint32_t)product * INV_P_MOCK;
    
    // === FIX OVERFLOW (Pastrat) ===
    #ifdef __CUDA_ARCH__
       unsigned __int128 t_full = (unsigned __int128)product + (unsigned __int128)m * P_MOCK;
       uint64_t t = (uint64_t)(t_full >> 32);
    #else
       // Castul la __int128 asigură că înmulțirea m * P nu pierde date înainte de adunare
       unsigned __int128 t_full = (unsigned __int128)product + (unsigned __int128)m * P_MOCK;
       uint64_t t = (uint64_t)(t_full >> 32);
    #endif
    
    if (t >= P_MOCK) return (field_t)(t - P_MOCK);
    return (field_t)t;
}

// === RIDICARE LA PUTERRE (Generică) ===

__host__ __device__ inline field_t pow_mod(field_t base, uint32_t exp, field_t mod) {
    field_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) {
             // Aici facem varianta clasică (non-Montgomery) pentru setup
             unsigned __int128 temp = (unsigned __int128)res * base;
             res = (field_t)(temp % mod);
        }
        unsigned __int128 temp = (unsigned __int128)base * base;
        base = (field_t)(temp % mod);
        exp /= 2;
    }
    return res;
}