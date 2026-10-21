#pragma once
#include <stdint.h>

/* ----- état ----- */
typedef struct {
  uint64_t s;            /* compteur */
  uint64_t gamma;        /* pas impair (lane) */
  uint64_t k0, k1, k2;   /* clés */
  int      r0, r1, r2;   /* rotations impaires */
} mmc_state_t;

/* API */
void     mmc_init(mmc_state_t* st, uint64_t seed, uint64_t lane);
double   mmc_next01(mmc_state_t* st);      /* double ∈ [0,1) */
uint64_t mmc_next_u64(mmc_state_t* st);    /* si tu veux la sortie 64b blanchie */
