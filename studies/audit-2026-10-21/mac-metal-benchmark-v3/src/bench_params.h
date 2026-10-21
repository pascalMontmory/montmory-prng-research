#ifndef MONTMORY_BENCH_PARAMS_H
#define MONTMORY_BENCH_PARAMS_H
#ifdef __METAL_VERSION__
typedef ulong BenchU64;
typedef uint BenchU32;
#else
#include <stdint.h>
typedef uint64_t BenchU64;
typedef uint32_t BenchU32;
#endif
struct BenchParams {
    BenchU64 seed, base, words, k0, k1, k2;
    BenchU32 r0, r1, r2, key0, key1;
    BenchU32 logical_lanes, draws_shift, physical_threads, output_shift, reserved;
};
#endif
