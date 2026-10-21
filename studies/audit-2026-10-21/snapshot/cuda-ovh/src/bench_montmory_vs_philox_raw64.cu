// src/bench_montmory_vs_philox_raw64.cu
// Benchmark Montmory GPU (64-bit raw) vs Philox4x32-10 (64-bit raw) sur A10

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// ----------------------------------------------------
// 1) Montmory GPU: même cœur que mmc_kernel.cu, mais version "raw 64 bits"
// ----------------------------------------------------

struct MmcState {
    unsigned long long s;
    unsigned long long gamma;
    unsigned long long k0, k1, k2;
    unsigned int       r0, r1, r2;
};

__device__ inline unsigned long long rotl64(unsigned long long x, unsigned int r) {
    r &= 63u;
    return (x << r) | (x >> ((64u - r) & 63u));
}

__device__ inline unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

__device__ inline unsigned long long pcg_dxsm(unsigned long long x) {
    unsigned long long w   = (x >> 32) ^ x;
    unsigned long long rot = x >> 59;
    w ^= (w >> 25);
    w *= 0xA24BAED496EE9B5FULL;
    w ^= (w >> 28);
    w *= 0x9FB21C651E98DF25ULL;
    w ^= (w >> 27);
    return (w >> rot) | (w << ((unsigned int)(-rot) & 63));
}

__device__ inline unsigned int mmc_f(unsigned long long x,
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

__device__ inline void mmc_init(MmcState& st,
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

__device__ inline unsigned long long perm3(unsigned long long x,
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

// Kernel Montmory: génère draws_per_thread tirages 64-bit par thread
__global__ void montmory_raw64_kernel(unsigned long long *out,
                                      unsigned long long seed,
                                      unsigned long long draws_per_thread)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long lane  = (unsigned long long)tid;

    MmcState st;
    mmc_init(st, seed, lane);

    unsigned long long ofs   = lane * draws_per_thread;
    unsigned long long start = lane * draws_per_thread; // base = 0

    // comme dans mmc_fill_kernel: aligner l'état sur base+lane*draws
    st.s = st.s + start * st.gamma;

    for (unsigned long long i = 0; i < draws_per_thread; ++i) {
        st.s += st.gamma;
        unsigned long long y = perm3(st.s, st);
        y = pcg_dxsm(y); // 64 bits bruts
        out[ofs + i] = y;
    }
}

// ----------------------------------------------------
// 2) Philox4x32-10: générateur 128-bit → 2x64-bit raw
// ----------------------------------------------------

struct philox_state_t {
    uint32_t c0, c1, c2, c3;
    uint32_t k0, k1;
};

__device__ inline uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t &hi) {
    unsigned long long p = (unsigned long long)a * (unsigned long long)b;
    hi = (uint32_t)(p >> 32);
    return (uint32_t)p;
}

__device__ inline void philox_round(philox_state_t &st) {
    const uint32_t M0 = 0xD2511F53U;
    const uint32_t M1 = 0xCD9E8D57U;
    uint32_t hi0, hi1;
    uint32_t lo0 = mulhilo32(M0, st.c0, hi0);
    uint32_t lo1 = mulhilo32(M1, st.c2, hi1);
    uint32_t t0 = hi1 ^ st.c1 ^ st.k0;
    uint32_t t1 = lo1;
    uint32_t t2 = hi0 ^ st.c3 ^ st.k1;
    uint32_t t3 = lo0;
    st.c0 = t0; st.c1 = t1; st.c2 = t2; st.c3 = t3;
}

__device__ inline void philox_bumpkey(philox_state_t &st) {
    const uint32_t W0 = 0x9E3779B9U;
    const uint32_t W1 = 0xBB67AE85U;
    st.k0 += W0;
    st.k1 += W1;
}

__device__ inline void philox_init_device(philox_state_t &st,
                                          unsigned long long global_seed,
                                          unsigned long long lane)
{
    st.c0 = (uint32_t)lane;
    st.c1 = (uint32_t)(lane >> 32);
    st.c2 = 0;
    st.c3 = 0;
    st.k0 = (uint32_t)global_seed;
    st.k1 = (uint32_t)(global_seed >> 32);
}

// Produit 2 x 64 bits par invocation
__device__ inline void philox_generate2(philox_state_t &st,
                                        unsigned long long &out0,
                                        unsigned long long &out1)
{
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        philox_round(st);
        philox_bumpkey(st);
    }
    out0 = ((uint64_t)st.c0 << 32) | st.c1;
    out1 = ((uint64_t)st.c2 << 32) | st.c3;

    uint64_t ctr_lo = ((uint64_t)st.c0) | ((uint64_t)st.c1 << 32);
    uint64_t ctr_hi = ((uint64_t)st.c2) | ((uint64_t)st.c3 << 32);
    ctr_lo += 1;
    if (ctr_lo == 0) ctr_hi += 1;
    st.c0 = (uint32_t)ctr_lo;
    st.c1 = (uint32_t)(ctr_lo >> 32);
    st.c2 = (uint32_t)ctr_hi;
    st.c3 = (uint32_t)(ctr_hi >> 32);
}

// Kernel Philox: génère draws_per_thread * 2 tirages 64-bit par thread
__global__ void philox_raw64_kernel(unsigned long long *out,
                                    unsigned long long seed,
                                    unsigned long long draws_per_thread)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long lane = (unsigned long long)tid;

    philox_state_t st;
    philox_init_device(st, seed, lane);

    unsigned long long ofs = lane * draws_per_thread * 2ULL;

    for (unsigned long long i = 0; i < draws_per_thread; ++i) {
        unsigned long long r0, r1;
        philox_generate2(st, r0, r1);
        out[ofs + 2ULL * i + 0] = r0;
        out[ofs + 2ULL * i + 1] = r1;
    }
}

// ----------------------------------------------------
// 3) Timing helpers : bench Montmory et Philox
// ----------------------------------------------------

static float run_montmory(int blocks, int threads,
                          unsigned long long seed,
                          unsigned long long draws_per_thread,
                          unsigned long long &total_samples,
                          unsigned long long &checksum)
{
    int total_threads = blocks * threads;
    total_samples = (unsigned long long)total_threads * draws_per_thread;

    unsigned long long *d_buf = nullptr;
    cudaMalloc((void**)&d_buf, total_samples * sizeof(unsigned long long));

    // warmup
    montmory_raw64_kernel<<<blocks, threads>>>(d_buf, seed, 1024);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    montmory_raw64_kernel<<<blocks, threads>>>(d_buf, seed, draws_per_thread);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // checksum
    unsigned long long *h_buf = (unsigned long long*)malloc(total_samples * sizeof(unsigned long long));
    cudaMemcpy(h_buf, d_buf, total_samples * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long acc = 0;
    for (unsigned long long i = 0; i < total_samples; ++i) {
        acc ^= h_buf[i];
    }
    checksum = acc;

    free(h_buf);
    cudaFree(d_buf);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

static float run_philox(int blocks, int threads,
                        unsigned long long seed,
                        unsigned long long draws_per_thread,
                        unsigned long long &total_samples,
                        unsigned long long &checksum)
{
    int total_threads = blocks * threads;
    total_samples = (unsigned long long)total_threads * draws_per_thread * 2ULL;

    unsigned long long *d_buf = nullptr;
    cudaMalloc((void**)&d_buf, total_samples * sizeof(unsigned long long));

    // warmup
    philox_raw64_kernel<<<blocks, threads>>>(d_buf, seed, 1024);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    philox_raw64_kernel<<<blocks, threads>>>(d_buf, seed, draws_per_thread);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    unsigned long long *h_buf = (unsigned long long*)malloc(total_samples * sizeof(unsigned long long));
    cudaMemcpy(h_buf, d_buf, total_samples * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long acc = 0;
    for (unsigned long long i = 0; i < total_samples; ++i) {
        acc ^= h_buf[i];
    }
    checksum = acc;

    free(h_buf);
    cudaFree(d_buf);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// ----------------------------------------------------
// 4) main : lance les deux benchs
// ----------------------------------------------------

int main() {
    const int BLOCKS  = 256;
    const int THREADS = 256; // 65536 threads
    const unsigned long long DRAWS_PER_THREAD = 1ULL << 14; // 16384

    const unsigned long long SEED = 0x123456789ABCDEF0ULL;

    unsigned long long total_samples, checksum;
    float ms;

    printf("CUDA PRNG benchmark (Montmory RAW64 vs Philox RAW64)\\n");
    printf("Blocks=%d, Threads=%d, Draws/thread=%llu\\n\\n",
           BLOCKS, THREADS, (unsigned long long)DRAWS_PER_THREAD);

    ms = run_montmory(BLOCKS, THREADS, SEED, DRAWS_PER_THREAD,
                      total_samples, checksum);
    {
        double ns_per_sample = (ms * 1e6) / (double)total_samples;
        double gbps = (double)total_samples * 8.0 / (ms * 1e6 * 1e3);
        printf("Montmory_RAW64   | samples=%llu | %8.3f ns/sample | %7.3f GB/s | checksum=0x%016llx\\n",
               (unsigned long long)total_samples,
               ns_per_sample,
               gbps,
               (unsigned long long)checksum);
    }

    ms = run_philox(BLOCKS, THREADS, SEED, DRAWS_PER_THREAD,
                    total_samples, checksum);
    {
        double ns_per_sample = (ms * 1e6) / (double)total_samples;
        double gbps = (double)total_samples * 8.0 / (ms * 1e6 * 1e3);
        printf("Philox4x32_10    | samples=%llu | %8.3f ns/sample | %7.3f GB/s | checksum=0x%016llx\\n",
               (unsigned long long)total_samples,
               ns_per_sample,
               gbps,
               (unsigned long long)checksum);
    }

    return 0;
}
