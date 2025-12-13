#include "verification.h"
#include <iostream>

// Constructor
Verifier::Verifier(int precision_bits) {
    this->precision = precision_bits;
    mpfr_set_default_prec(precision);
}

// Destructor
Verifier::~Verifier() {
    mpfr_free_cache();
}

// Implementarea NTT Naiv pe CPU (Golden Model)
std::vector<field_t> Verifier::compute_reference_ntt_naive(const std::vector<field_t>& input, field_t omega, field_t mod) {
    size_t N = input.size();
    std::vector<field_t> output(N);

    // DFT Naiv: Două bucle for
    for (size_t k = 0; k < N; k++) {
        field_t sum = 0;
        
        // Calculăm w_k = omega^k
        field_t w_k = pow_mod(omega, k, mod); 
        
        field_t current_w = 1; // w^(j*k) începe de la 1

        for (size_t j = 0; j < N; j++) {
            // term = input[j] * w^(j*k)
            
            // Folosim __int128 pentru a preveni overflow la înmulțire pe CPU
            unsigned __int128 term = (unsigned __int128)input[j] * current_w;
            
            // sum = (sum + term) % mod
            sum = add_mod(sum, (field_t)(term % mod)); // Folosim add_mod din montgomery.h sau adunare simplă cu %

            // Update w^(j*k) pentru următorul pas
            unsigned __int128 next_w = (unsigned __int128)current_w * w_k;
            current_w = (field_t)(next_w % mod);
        }
        output[k] = sum;
    }
    return output;
}