#include <metal_stdlib>
using namespace metal;

struct Params {
  ulong seed;
  ulong draws_per_thread;
};

static inline ulong rotl64(ulong x, uint r){ r&=63u; return (x<<r)|(x>>((64u-r)&63u)); }
static inline ulong splitmix64(ulong x){
  x += 0x9E3779B97F4A7C15UL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
  return x ^ (x >> 31);
}
static inline ulong pcg_dxsm(ulong x){
  ulong w = (x>>32) ^ x;
  ulong rot = x >> 59;
  w ^= (w>>25); w *= 0xA24BAED496EE9B5FUL;
  w ^= (w>>28); w *= 0x9FB21C651E98DF25UL;
  w ^= (w>>27);
  return (w>>rot) | (w<<((-rot)&63));
}
static inline uint mmc_f(ulong x, ulong k, uint r0, uint r1){
  ulong y = (x & 1UL) ? (rotl64(x*3UL + k, r0) ^ (x>>1))
                      : (rotl64(x + k,    r1) ^ (x*5UL));
  return (uint)(y & 0xFFFFFFFFUL);
}
struct MmcState {
  ulong s, gamma, k0, k1, k2;
  uint  r0, r1, r2;
};
static inline void mmc_init(thread MmcState& st, ulong seed, ulong lane){
  st.s = seed;
  st.gamma = (0x9E3779B97F4A7C15UL + (lane<<1)) | 1UL;
  st.k0 = splitmix64(seed);
  st.k1 = splitmix64(st.k0);
  st.k2 = splitmix64(st.k1);
  ulong r = splitmix64(st.k2);
  st.r0 = ((uint)( r       & 63u)) | 1u;
  st.r1 = ((uint)((r>> 8 ) & 63u)) | 1u;
  st.r2 = ((uint)((r>>16 ) & 63u)) | 1u;
}
static inline ulong mmc_perm3(ulong x, thread const MmcState& st){
  ulong L=(x>>32)&0xFFFFFFFFUL, R=x&0xFFFFFFFFUL, t; uint F;
  F=mmc_f(R, st.k0, st.r0, st.r1); L^=F; t=L; L=R; R=t;
  F=mmc_f(R, st.k1, st.r1, st.r2); L^=F; t=L; L=R; R=t;
  F=mmc_f(R, st.k2, st.r2, st.r0); L^=F; t=L; L=R; R=t;
  return ((L & 0xFFFFFFFFUL)<<32) | (R & 0xFFFFFFFFUL);
}

kernel void mmc_fill(device double* out [[buffer(0)]],
                     constant Params& P [[buffer(1)]],
                     uint tid [[thread_position_in_grid]])
{
  MmcState st; mmc_init(st, P.seed, (ulong)tid);
  ulong base = (ulong)tid * P.draws_per_thread;
  for (ulong i=0; i<P.draws_per_thread; ++i){
    st.s += st.gamma;
    ulong y = mmc_perm3(st.s, st);
    y = pcg_dxsm(y);
    ulong mant = (y >> 11) & ((1UL<<53)-1UL);
    out[base + i] = (double)mant / (double)(1UL<<53);
  }
}
