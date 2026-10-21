/* Portable golden-vector emitter: compile with the frozen src/mmc_core.c.
 * Text is fixed-width hexadecimal, independent of byte order and printf widths.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include "mmc_core.c"
int main(void) {
    const uint64_t seeds[] = {0,1,42,UINT64_MAX,UINT64_C(0x8000000000000000)};
    const uint64_t lanes[] = {0,1,2,8191,UINT64_C(0xffffffff)};
    const uint64_t bases[] = {0,UINT64_C(0xffffffff),UINT64_C(0x8000000000000000),UINT64_MAX-15};
    for (size_t a=0;a<5;++a) for(size_t b=0;b<5;++b) for(size_t c=0;c<4;++c) {
        mmc_state_t st;
        mmc_init(&st,seeds[a],lanes[b]);
        st.s += bases[c]*st.gamma;
        for(unsigned int i=0;i<16;++i) {
            uint64_t word=mmc_next_u64(&st);
            printf("%016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %014" PRIx64 "\n",
                   seeds[a],lanes[b],bases[c]+i,word,word>>11);
        }
    }
    return ferror(stdout) ? 1 : 0;
}
