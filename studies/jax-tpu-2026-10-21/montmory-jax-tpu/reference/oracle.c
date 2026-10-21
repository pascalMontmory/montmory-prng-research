/* Independent scalar uint64 reference: original core is included unchanged. */
#include <stddef.h>
#include "mmc_core.c"

void reference_words(uint64_t seed, const uint64_t *indices, uint64_t *out, size_t n) {
    mmc_state_t state;
    mmc_init(&state, seed, 0);
    for (size_t i=0; i<n; ++i) {
        const uint64_t k=indices[i], lane=(k/4096)%2048;
        state.gamma=UINT64_C(0x9E3779B97F4A7C15)+2*lane;
        state.s=seed+k*state.gamma;
        out[i]=mmc_next_u64(&state);
    }
}

void reference_lanes(const uint64_t *seeds, const uint64_t *indices,
                     const uint64_t *lanes, uint64_t *out, size_t n) {
    for(size_t i=0;i<n;++i) {
        mmc_state_t state;
        mmc_init(&state,seeds[i],lanes[i]);
        state.s+=indices[i]*state.gamma;
        out[i]=mmc_next_u64(&state);
    }
}
