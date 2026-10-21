/* montmory_prng.c — Montmory–Collatz (3 tours Feistel) + DXSM whitening + wrapper TestU01
   - Cœur: compteur affine 64-bit (s += gamma, gamma impair par lane)
   - Permutation: 3 tours Feistel, round f() dérivé de la logique Collatz (pair/impair) + rotations impaires
   - Clés/rotations dérivées via splitmix64(seed)
   - Sortie: DXSM( permute64(s) ) -> extraction 53 bits de poids fort -> double ∈ [0,1)
*/

#include <stdint.h>
#include <stdio.h>      /* pour sscanf */
#include "unif01.h"
#include "bbattery.h"

/* ========== utilitaires ==========
   - rotl64: rotation gauche sur 64 bits
   - splitmix64: expand un seed en séquence pseudo-aléatoire pour dériver k0,k1,k2,r0,r1,r2
*/
static inline uint64_t rotl64(uint64_t x, unsigned r) {
  r &= 63u;
  return (x << r) | (x >> (64u - r));
}
static inline uint64_t splitmix64(uint64_t x) {
  x += UINT64_C(0x9E3779B97F4A7C15);
  x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
  return x ^ (x >> 31);
}

/* ========== DXSM/PCG-style whitening (très léger, robuste) ========== */
static inline uint64_t pcg_dxsm(uint64_t x) {
  uint64_t word = (x >> 32) ^ x;                 /* mix hi/lo */
  uint64_t rot  = x >> 59;                       /* 5 bits */
  word ^= (word >> 25);
  word *= UINT64_C(0xA24B3F6F);
  word ^= (word >> 28);
  word *= UINT64_C(0x9FB21C651E98DF25);
  word ^= (word >> 27);
  return (word >> rot) | (word << ((-rot) & 63));
}

/* ========== état du PRNG ========== */
typedef struct {
  uint64_t s;            /* compteur 64 bits */
  uint64_t gamma;        /* incrément impair (par lane) */
  uint64_t k0, k1, k2;   /* clés pour f() aux 3 tours */
  int      r0, r1, r2;   /* rotations impaires (1..31 ou 1..63 ; on force le bit0) */
} mmc_state_t;

/* ========== fonction f() Collatz-like sur 32 bits ========== */
static inline uint32_t mmc_f(uint64_t x, uint64_t k, int r0, int r1) {
  /* branche pair/impair façon Collatz + rotations pour diffusion */
  uint64_t y;
  if (x & 1ULL)  {                      /* impair */
    y = (rotl64(x * 3ULL + k, (unsigned)r0) ^ (x >> 1));
  } else {                              /* pair */
    y = (rotl64(x + k,        (unsigned)r1) ^ (x * 5ULL));
  }
  return (uint32_t)(y & 0xFFFFFFFFULL);
}

/* ========== permutation 64-bit : 3 tours Feistel ========== */
static inline uint64_t mmc_permute64(uint64_t x, const mmc_state_t* st) {
  uint64_t L = (x >> 32) & 0xFFFFFFFFULL;
  uint64_t R =  x        & 0xFFFFFFFFULL;
  uint64_t t;
  uint32_t F;

  /* Tour 0 */
  F = mmc_f(R, st->k0, st->r0, st->r1);
  L ^= F;  t = L;  L = R;  R = t;

  /* Tour 1 */
  F = mmc_f(R, st->k1, st->r1, st->r2);
  L ^= F;  t = L;  L = R;  R = t;

  /* Tour 2 */
  F = mmc_f(R, st->k2, st->r2, st->r0);
  L ^= F;  t = L;  L = R;  R = t;

  return ((L & 0xFFFFFFFFULL) << 32) | (R & 0xFFFFFFFFULL);
}

/* ========== initialisation (seed + lane) ========== */
static inline void mmc_init(mmc_state_t *st, uint64_t seed, uint64_t lane) {
  st->s     = seed;
  /* gamma impair, dérivé d’une constante “dorée” + lane, pour indépendance des lanes */
  st->gamma = (UINT64_C(0x9E3779B97F4A7C15) + (lane << 1)) | 1ULL;

  /* dérive 3 clés et 3 rotations impaires via splitmix64 */
  uint64_t k0 = splitmix64(seed);
  uint64_t k1 = splitmix64(k0);
  uint64_t k2 = splitmix64(k1);
  st->k0 = k0;  st->k1 = k1;  st->k2 = k2;

  uint64_t r = splitmix64(k2);
  /* forcer des décalages impairs entre 1 et 63 (low-bit = 1) */
  st->r0 = (int)(( r        & 63u) | 1u);
  st->r1 = (int)(((r >> 8 ) & 63u) | 1u);
  st->r2 = (int)(((r >> 16) & 63u) | 1u);
}

/* ========== tirage d’un double ∈ [0,1) ========== */
static inline double mmc_next01(mmc_state_t *st) {
  st->s += st->gamma;                                 /* compteur 64-bit */
  uint64_t y = mmc_permute64(st->s, st);              /* permutation Collatz 3 tours */
  y = pcg_dxsm(y);                                    /* whitening DXSM */
  uint64_t mant = (y >> 11) & ((1ULL << 53) - 1ULL);  /* 53 bits de mantisse */
  return (double)mant / (double)(1ULL << 53);
}

/* ========== Wrapper TestU01 ========== */
static mmc_state_t G;          /* état global */
static double wrap(void){      /* signature attendue: double(void) */
  return mmc_next01(&G);
}

int main(int argc, char **argv){
  uint64_t seed = 42ULL;
  uint64_t lane = 0ULL;
  if (argc >= 2) sscanf(argv[1], "%llu", &seed);
  if (argc >= 3) sscanf(argv[2], "%llu", &lane);

  mmc_init(&G, seed, lane);

  unif01_Gen *gen = unif01_CreateExternGen01("Montmory-REAL-MMC-3R+DXSM", wrap);

  int doCrush = 0, doBig = 0;
  if (argc >= 4){
    if      (argv[3][0]=='c' || argv[3][0]=='C') doCrush = 1;
    else if (argv[3][0]=='b' || argv[3][0]=='B') doBig   = 1;
  }
  if (doBig)        bbattery_BigCrush (gen);
  else if (doCrush) bbattery_Crush    (gen);
  else              bbattery_SmallCrush(gen);

  unif01_Gen *g = gen; (void)g;  /* silence unused warning if needed */
  unif01_DeleteExternGen01(gen);
  return 0;
}
