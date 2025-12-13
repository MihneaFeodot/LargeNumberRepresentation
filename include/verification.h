#pragma once
#include <vector>
#include <cstdint>
#include <mpfr.h> 
#include "montgomery.h" 

class Verifier {
private:
    mpfr_prec_t precision;

public:
    Verifier(int precision_bits);
    ~Verifier();
    std::vector<field_t> compute_reference_ntt_naive(const std::vector<field_t>& input, field_t omega, field_t mod);
};
