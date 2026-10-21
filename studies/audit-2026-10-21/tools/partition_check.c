/* Check random access and physical worker independence without changing the
 * recovered core or its legacy logical-lane ordering. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "mmc_core.c"

static uint64_t word_at(uint64_t seed, uint64_t index,
                        uint64_t logical_threads, uint64_t logical_draws) {
    uint64_t lane = (index / logical_draws) % logical_threads;
    mmc_state_t st;
    mmc_init(&st, seed, lane);
    st.s += index * st.gamma;
    return mmc_next_u64(&st);
}

int main(void) {
    enum { N = 100003 };
    uint64_t *expected = malloc(N * sizeof(*expected));
    uint64_t *actual = malloc(N * sizeof(*actual));
    if (!expected || !actual) return 2;
    const uint64_t seeds[] = {0, 42, UINT64_MAX};
    const uint64_t configs[][2] = {{1,1},{7,11},{2048,4096},{8192,16384}};
    uint64_t comparisons=0;
    unsigned int teams=0;
    for (size_t s=0;s<3;++s) for(size_t c=0;c<4;++c) {
        for (uint64_t i=0;i<N;++i) expected[i]=word_at(seeds[s],i,configs[c][0],configs[c][1]);
        for(int workers=1;workers<=4;workers*=2) {
            int observed=0;
            #pragma omp parallel num_threads(workers) shared(observed)
            {
                #pragma omp single
                observed=omp_get_num_threads();
                #pragma omp for schedule(dynamic,17)
                for(uint64_t j=0;j<N;++j) {
                    /* Reverse global order to exercise scheduling independence. */
                    uint64_t i=N-1-j;
                    actual[i]=word_at(seeds[s],i,configs[c][0],configs[c][1]);
                }
            }
            if(observed!=workers || memcmp(expected,actual,N*sizeof(*expected))) return 1;
            teams |= 1u << observed;
            comparisons += N;
        }
    }
    printf("{\"status\":\"pass\",\"word_comparisons\":%" PRIu64 ",\"actual_worker_counts\":[1,2,4],\"team_mask\":%u}\n",comparisons,teams);
    free(expected); free(actual);
    return 0;
}
