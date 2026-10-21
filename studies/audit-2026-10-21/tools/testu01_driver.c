/* Audit adapter for the frozen legacy source. No change to the output mixer.
 * metal01 reproduces the recovered kernel's ordered stream on a CPU;
 * it does not exercise Metal/CUDA hardware or the Objective-C++ wrapper.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "unif01.h"
#include "bbattery.h"
#include "mmc_core.c"

static mmc_state_t state;
static uint64_t seed_value, position, threads, draws;
static int metal_order;

static uint64_t next_word(void) {
    if (metal_order && position % draws == 0) {
        uint64_t lane = (position / draws) % threads;
        state.gamma = UINT64_C(0x9E3779B97F4A7C15) + 2 * lane;
        state.s = seed_value + position * state.gamma;
    }
    ++position;
    return mmc_next_u64(&state);
}
static double next01(void) { return (double)(next_word() >> 11) * 0x1p-53; }
static unsigned int low32(void) { return (unsigned int)(uint32_t)next_word(); }
static unsigned int high32(void) { return (unsigned int)(next_word() >> 32); }

static uint64_t number(const char *text) {
    char *end;
    errno = 0;
    if (*text == '-' || *text == '+') { fprintf(stderr,"Invalid unsigned integer\n"); exit(2); }
    unsigned long long n = strtoull(text, &end, 10);
    if (errno || *end || end == text) { fprintf(stderr,"Invalid integer\n"); exit(2); }
    return (uint64_t)n;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr,"usage: %s cpu01|metal01|low32|high32 seed threads draws small|crush|big\n",argv[0]);
        return 2;
    }
    seed_value = number(argv[2]); threads = number(argv[3]); draws = number(argv[4]);
    if (!threads || !draws || threads > UINT32_MAX || draws > UINT32_MAX) return 2;
    metal_order = strcmp(argv[1],"metal01") == 0;
    mmc_init(&state,seed_value,0);
    unif01_Gen *gen;
    if (!strcmp(argv[1],"low32")) gen=unif01_CreateExternGenBits("MMC legacy low32 audit",low32);
    else if (!strcmp(argv[1],"high32")) gen=unif01_CreateExternGenBits("MMC legacy high32 audit",high32);
    else if (metal_order || !strcmp(argv[1],"cpu01")) gen=unif01_CreateExternGen01("MMC legacy audit CPU replay",next01);
    else return 2;
    printf("audit mode=%s seed=%llu threads=%llu draws=%llu battery=%s\n",
           argv[1],(unsigned long long)seed_value,(unsigned long long)threads,
           (unsigned long long)draws,argv[5]);
    if (!strcmp(argv[5],"small")) bbattery_SmallCrush(gen);
    else if (!strcmp(argv[5],"crush")) bbattery_Crush(gen);
    else if (!strcmp(argv[5],"big")) bbattery_BigCrush(gen);
    else return 2;
    if (!strcmp(argv[1],"low32") || !strcmp(argv[1],"high32")) unif01_DeleteExternGenBits(gen);
    else unif01_DeleteExternGen01(gen);
    return 0;
}
