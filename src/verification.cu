#include "verification.h"
#include <iostream>

Verifier::Verifier(int precision_bits) {
    this->precision = precision_bits;
    mpfr_set_default_prec(precision);
}

Verifier::~Verifier() {
    mpfr_free_cache();
}

std::vector<field_t> Verifier::compute_reference_ntt_naive(const std::vector<field_t>& input, field_t omega, field_t mod) {
    size_t N = input.size();
    std::vector<field_t> output(N);

    for (size_t k = 0; k < N; k++) {
        field_t sum = 0;
        field_t w_k = pow_mod(omega, k, mod); 
        field_t current_w = 1; 

        for (size_t j = 0; j < N; j++) {
            unsigned __int128 term = (unsigned __int128)input[j] * current_w;
            sum = add_mod(sum, (field_t)(term % mod)); 

            unsigned __int128 next_w = (unsigned __int128)current_w * w_k;
            current_w = (field_t)(next_w % mod);
        }
        output[k] = sum;
    }
    return output;
}
