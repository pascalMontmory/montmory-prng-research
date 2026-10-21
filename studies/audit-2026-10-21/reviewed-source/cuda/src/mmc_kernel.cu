// mmc_kernel.cu
#include "mmc_cuda_params.h"

// ----- Types compatibles Metal → CUDA -----

struct MmcState {
    unsigned long long s;
    unsigned long long gamma;
    unsigned long long k0, k1, k2;
    unsigned int       r0, r1, r2;
};

// ----- Fonctions inline (même logique que Metal) -----
__host__ __device__ inline unsigned long long rotl64(unsigned long long x, unsigned int r) {
    r &= 63u;
    return (x << r) | (x >> ((64u - r) & 63u));
}

__host__ __device__ inline unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

__host__ __device__ inline unsigned long long pcg_dxsm(unsigned long long x) {
    unsigned long long w   = (x >> 32) ^ x;
    unsigned long long rot = x >> 59;
    w ^= (w >> 25);
    w *= 0xA24BAED496EE9B5FULL;
    w ^= (w >> 28);
    w *= 0x9FB21C651E98DF25ULL;
    w ^= (w >> 27);
    return (w >> rot) | (w << ((unsigned int)(-rot) & 63));
}

__host__ __device__ inline unsigned int mmc_f(unsigned long long x,
                                              unsigned long long k,
                                              unsigned int r0,
                                              unsigned int r1)
{
    unsigned long long y;
    if (x & 1ULL) {
        y = rotl64(x * 3ULL + k, r0) ^ (x >> 1);
    } else {
        y = rotl64(x + k, r1) ^ (x * 5ULL);
    }
    return (unsigned int)(y & 0xFFFFFFFFULL);
}

__host__ __device__ inline void mmc_init(MmcState& st,
                                         unsigned long long seed,
                                         unsigned long long lane)
{
    st.s      = seed;
    st.gamma  = (0x9E3779B97F4A7C15ULL + (lane << 1)) | 1ULL;
    st.k0     = splitmix64(seed);
    st.k1     = splitmix64(st.k0);
    st.k2     = splitmix64(st.k1);
    unsigned long long r = splitmix64(st.k2);
    st.r0 = ((unsigned int)( ( r      ) & 63u)) | 1u;
    st.r1 = ((unsigned int)((r >>  8) & 63u)) | 1u;
    st.r2 = ((unsigned int)((r >> 16) & 63u)) | 1u;
}

__host__ __device__ inline unsigned long long perm3(unsigned long long x,
                                                    const MmcState& st)
{
    unsigned long long L = (x >> 32) & 0xFFFFFFFFULL;
    unsigned long long R =  x        & 0xFFFFFFFFULL;
    unsigned long long t;
    unsigned int F;

    F = mmc_f(R, st.k0, st.r0, st.r1);
    L ^= F; t = L; L = R; R = t;

    F = mmc_f(R, st.k1, st.r1, st.r2);
    L ^= F; t = L; L = R; R = t;

    F = mmc_f(R, st.k2, st.r2, st.r0);
    L ^= F; t = L; L = R; R = t;

    return ((L & 0xFFFFFFFFULL) << 32) | (R & 0xFFFFFFFFULL);
}

// ----- Kernel CUDA équivalent à mmc_fill (Metal) -----
// out : buffer device où l'on écrit les mantissas 53 bits
// P   : paramètres (seed, base, draws) passés par valeur
extern "C"
__global__ void mmc_fill_kernel(uint64_t* out, Params P)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= P.lanes) return;
    unsigned long long lane = (unsigned long long)tid;

    MmcState st;
    mmc_init(st, P.seed, lane);

    unsigned long long draws = P.draws;
    unsigned long long ofs   = lane * draws;
    unsigned long long start = P.base + lane * draws;

    // aligner l'état sur base+lane*draws comme en Metal
    st.s = st.s + start * st.gamma;

    for (unsigned long long i = 0; i < draws; ++i) {
        st.s += st.gamma;
        unsigned long long y = perm3(st.s, st);
        y = pcg_dxsm(y);
        unsigned long long mant = (y >> 11) & ((1ULL << 53) - 1ULL);
        out[ofs + i] = mant;
    }
}
