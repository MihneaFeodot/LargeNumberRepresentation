# 🚀 LargeNumberRepresentation

## Aritmetica Numerelor Mari (BigInt) și Transformata Numerică Teoretică (NTT) pe CUDA

Acest proiect implementează o **bibliotecă de aritmetică BigInt de înaltă performanță**, optimizată pentru **GPU-uri NVIDIA CUDA**, având ca obiectiv principal **înmulțirea rapidă a numerelor mari și a polinoamelor** folosind **Transformata Numerică Teoretică (NTT)** în complexitate (O(N \log N)).

Scopul este demonstrarea unui **speedup de 5x–10x** față de implementările clasice pe CPU, prin exploatarea paralelismului masiv al GPU-ului și a optimizărilor avansate de memorie.

---

## ✨ Caracteristici Principale

* Reprezentare **BigInt** bazată pe limb-uri de 32 de biți
* Aritmetică paralelă pe GPU (adunare, scădere)
* Înmulțire modulară extrem de rapidă folosind **Montgomery Multiplication**
* Implementare completă **NTT (Cooley–Tukey)** pe CUDA
* Optimizări de memorie (Shared Memory, Bank Conflict Avoidance)
* Verificare riguroasă a corectitudinii cu **MPFR (Golden Model)**

---

## 👥 Structura Echipei și Responsabilități

| Membru       | Rol                     | Fișiere Cheie                                      | Contribuții                                                         |
| ------------ | ----------------------- | -------------------------------------------------- | ------------------------------------------------------------------- |
| **Membru 1** | Infrastructure & BigInt | `bigint.cuh`, `bigint_impl.cu`, `bigint_utils.cpp` | Structura BigInt, management memorie CUDA, adunare/scădere paralelă |
| **Membru 2** | Algoritm NTT            | `ntt_kernel.cuh`, `montgomery.h`                   | Bit-reversal, butterfly kernels, Cooley–Tukey                       |
| **Membru 3** | Optimizare & Verificare | `montgomery.h`, `verification.cpp`                 | Montgomery multiplication, optimizare memorie, validare MPFR        |

---

## 🧱 Reprezentarea BigInt

Numerele mari sunt reprezentate ca vectori de limb-uri pe 32 de biți, pentru a permite operații intermediare sigure pe 64/128 de biți.

```cpp
typedef uint32_t limb_t;
typedef uint64_t double_limb_t;

struct BigInt {
    limb_t* limbs;        // vector de limb-uri
    size_t num_limbs;     // număr de limb-uri
    bool is_negative;     // semn
    bool on_device;       // flag host/device
};
```

---

## ➕➖ Aritmetică Paralelă (Adunare / Scădere)

Propagarea carry/borrow este realizată în **doi pași**, folosind o abordare de tip *parallel scan*:

1. **Calcul local** – fiecare thread calculează suma și generează un carry local
2. **Propagare globală** – carry-urile sunt propagate într-un kernel separat

Această strategie elimină dependențele secvențiale și permite scalarea pe GPU.

---

## ⚡ Aritmetică Modulară – Montgomery Multiplication

Pentru a evita operațiile costisitoare de modulo, NTT folosește **înmulțirea Montgomery** cu baza:

* (R = 2^{32})
* Modul prim: `P = 3221225473 (0xC0000001)`

Implementarea este sigură la overflow prin utilizarea tipului `unsigned __int128`.

---

## 🧠 Optimizarea Memoriei GPU

Pentru performanță maximă în Shared Memory:

* Se evită **bank conflicts** prin indexare cu padding

```cpp
#define PADDED_INDEX(i) ((i) + ((i) >> 5))
```

Această tehnică este esențială în etapele **butterfly** ale NTT.

---

## 🔄 Transformata Numerică Teoretică (NTT)

Implementarea urmează algoritmul **Cooley–Tukey**:

1. **Bit-Reversal Permutation** – reordonarea inițială a elementelor
2. **Etape Butterfly** – calcul paralel folosind Montgomery multiplication

Fiecare etapă este lansată ca un kernel CUDA, utilizând Shared Memory cu padding.

---

## ✅ Verificare și Corectitudine

Pentru a garanta rezultatele:

* Implementare de referință pe CPU ((O(N^2)))
* Folosirea bibliotecilor **MPFR** și **GMP**
* Compararea rezultatelor GPU cu Golden Model

Această etapă asigură corectitudine absolută, chiar și pentru cazuri limită.

---

## ⚙️ Compilare

### Dependențe

* NVIDIA CUDA Toolkit
* Compilator C++ cu suport `__int128`
* Bibliotecile **MPFR** și **GMP**

### Comandă de Compilare

```bash
nvcc -std=c++17 -o ntt_bigint \
     bigint_impl.cu \
     bigint_utils.cpp \
     verification.cpp \
     main.cpp \
     -lmpfr -lgmp
```

---

## ▶️ Rulare

```bash
./ntt_bigint
```

---

## 📈 Rezultate Așteptate

* Speedup semnificativ față de CPU (5x–10x)
* Scalare eficientă pentru dimensiuni mari
* Precizie matematică garantată

---

## 📌 Note Finale

Acest proiect demonstrează cum **aritmetica numerelor mari** și **algoritmii de tip FFT/NTT** pot beneficia masiv de paralelismul GPU, fiind aplicabili în:

* Criptografie
* Calcul simbolic
* Sisteme de algebră computațională
* High‑Performance Computing (HPC)

---

🧑‍💻 *Proiect realizat în scop educațional și experimental, cu accent pe performanță și corectitudine matematică.*
