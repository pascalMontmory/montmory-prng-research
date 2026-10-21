#ifndef MMC_CUDA_PARAMS_H
#define MMC_CUDA_PARAMS_H
#include <stdint.h>

// Shared definition: wrapper and kernel must agree on the parameter type.
struct Params {
    uint64_t seed;
    uint64_t base;
    uint64_t draws;
    uint32_t lanes;
};

#endif
