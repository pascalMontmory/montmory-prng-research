// mmc_gpu_wrap.cu
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

//
//  Ce fichier remplace ton mmc_gpu_wrap.mm (Metal) par un wrapper CUDA.
//  Il gère :
//    - l’allocation GPU
//    - l’appel du kernel CUDA mmc_fill_kernel()
//    - la copie device→host
//    - un ring-buffer CPU comme dans la version Metal
//
//  API publique identique :
//    extern "C" void   mmc_gpu_init(uint64_t seed, uint32_t threads, uint32_t draws);
//    extern "C" double mmc_gpu_next01(void);
//

// ----------------------------
//  Structures & kernel extern
// ----------------------------
struct Params {
    uint64_t seed;
    uint64_t base;
    uint64_t draws;
};

extern "C"
__global__ void mmc_fill_kernel(uint64_t *out, Params P);

// ----------------------------
//  Variables globales (comme Metal)
// ----------------------------
static uint64_t              gSeed     = 42;
static uint64_t              gBase     = 0;
static uint32_t              gThreads  = 2048;
static uint32_t              gDpt      = 4096;

static uint64_t             *gDevBuf   = nullptr;   // buffer GPU des mantissas 53 bits
static std::vector<double>   gRing;                 // ring buffer CPU
static size_t                gHead = 0, gTail = 0, gCap = 0;

// ----------------------------
//  Recharge le ring depuis le GPU
// ----------------------------
static void refillGPU()
{
    uint64_t N = (uint64_t)gThreads * (uint64_t)gDpt;

    Params P;
    P.seed  = gSeed;
    P.base  = gBase;
    P.draws = gDpt;

    // configuration du kernel
    dim3 block(256);
    dim3 grid((gThreads + block.x - 1) / block.x);

    // appel GPU
    mmc_fill_kernel<<<grid, block>>>(gDevBuf, P);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // copie device -> host
    std::vector<uint64_t> tmp(N);
    err = cudaMemcpy(tmp.data(), gDevBuf,
                     N * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // conversion mantissa → double [0,1)
    const double inv2_53 = 1.0 / 9007199254740992.0;

    for (uint64_t i = 0; i < N; ++i) {
        gRing[gHead] = (double)tmp[i] * inv2_53;
        gHead = (gHead + 1) % gCap;
        if (gHead == gTail) {
            gTail = (gTail + 1) % gCap; // écrase l’élément le plus ancien
        }
    }

    gBase += N;
}

// ----------------------------
//  API PUBLIQUE (comme dans Metal)
// ----------------------------

extern "C"
void mmc_gpu_init(uint64_t seed,
                  uint32_t threads,
                  uint32_t draws_per_thread)
{
    gSeed    = seed;
    gThreads = threads;
    gDpt     = draws_per_thread;
    gBase    = 0;

    uint64_t N = (uint64_t)threads * (uint64_t)draws_per_thread;

    // allocation ring CPU
    gCap = 2 * N;
    gRing.assign(gCap, 0.0);
    gHead = gTail = 0;

    // allocation GPU
    cudaError_t err = cudaMalloc((void**)&gDevBuf,
                                 N * sizeof(uint64_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // premier remplissage
    refillGPU();
}

extern "C"
double mmc_gpu_next01(void)
{
    if (gTail == gHead) {
        refillGPU();
    }
    double v = gRing[gTail];
    gTail = (gTail + 1) % gCap;
    return v;
}
