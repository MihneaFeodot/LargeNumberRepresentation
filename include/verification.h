#pragma once
#include <vector>
#include <cstdint>
#include <mpfr.h> // Necesită biblioteca MPFR
#include "montgomery.h" // Pentru tipul field_t

class Verifier {
private:
    mpfr_prec_t precision;

public:
    // Constructor: Setează precizia (nr de biți)
    Verifier(int precision_bits);

    // Destructor: Curăță resursele
    ~Verifier();

    // Calculează NTT-ul de referință pe CPU (încet, dar corect)
    // Folosește algoritmul Naiv O(N^2) pentru a fi sigur de rezultat
    std::vector<field_t> compute_reference_ntt_naive(const std::vector<field_t>& input, field_t omega, field_t mod);
};