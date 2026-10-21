#pragma once
#include <stdint.h>
typedef struct { uint64_t s,gamma,k0,k1,k2; int r0,r1,r2; } mmc_state_t;
void     mmc_init(mmc_state_t* st, uint64_t seed, uint64_t lane);
double   mmc_next01(mmc_state_t* st);
uint64_t mmc_next_u64(mmc_state_t* st);
