#ifndef MMC_CORE_H
#define MMC_CORE_H

#include <stdint.h>

/* État du PRNG mmc — doit être cohérent avec MmcState côté CUDA */
typedef struct {
    uint64_t s;
    uint64_t gamma;
    uint64_t k0, k1, k2;
    int      r0, r1, r2;
} mmc_state_t;

/* Initialisation (seed + lane) */
void     mmc_init(mmc_state_t* st, uint64_t seed, uint64_t lane);
/* Prochain 64 bits du PRNG */
uint64_t mmc_next_u64(mmc_state_t* st);
/* Prochain double U[0,1) avec mantisse 53 bits */
double   mmc_next01(mmc_state_t* st);

#endif /* MMC_CORE_H */
