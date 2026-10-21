#include "mmc_core.h"

/* utils */
static inline uint64_t rotl64(uint64_t x, unsigned r){ r&=63u; return (x<<r)|(x>>((64u-r)&63u)); }
static inline uint64_t splitmix64(uint64_t x){
  x+=UINT64_C(0x9E3779B97F4A7C15);
  x=(x^(x>>30))*UINT64_C(0xBF58476D1CE4E5B9);
  x=(x^(x>>27))*UINT64_C(0x94D049BB133111EB);
  return x^(x>>31);
}
/* DXSM / PCG-like */
static inline uint64_t pcg_dxsm(uint64_t x){
  uint64_t w=(x>>32)^x, rot=x>>59;
  w^=w>>25; w*=UINT64_C(0xA24BAED496EE9B5F);
  w^=w>>28; w*=UINT64_C(0x9FB21C651E98DF25);
  w^=w>>27;
  return (w>>rot)|(w<<((-rot)&63));
}
/* f Collatz-like 32b */
static inline uint32_t mmc_f(uint64_t x, uint64_t k, int r0, int r1){
  uint64_t y = (x & 1ULL) ? (rotl64(x*3ULL + k, (unsigned)r0) ^ (x>>1))
                          : (rotl64(x + k,      (unsigned)r1) ^ (x*5ULL));
  return (uint32_t)(y & 0xFFFFFFFFULL);
}
/* 3 tours Feistel */
static inline uint64_t mmc_permute64(uint64_t x, const mmc_state_t* st){
  uint64_t L=(x>>32)&0xFFFFFFFFULL, R=x&0xFFFFFFFFULL, t; uint32_t F;
  F=mmc_f(R, st->k0, st->r0, st->r1); L^=F; t=L; L=R; R=t;
  F=mmc_f(R, st->k1, st->r1, st->r2); L^=F; t=L; L=R; R=t;
  F=mmc_f(R, st->k2, st->r2, st->r0); L^=F; t=L; L=R; R=t;
  return ((L & 0xFFFFFFFFULL)<<32) | (R & 0xFFFFFFFFULL);
}
/* init */
void mmc_init(mmc_state_t* st, uint64_t seed, uint64_t lane){
  st->s = seed;
  st->gamma = (UINT64_C(0x9E3779B97F4A7C15) + (lane<<1)) | 1ULL; /* impair */
  uint64_t k0=splitmix64(seed), k1=splitmix64(k0), k2=splitmix64(k1);
  st->k0=k0; st->k1=k1; st->k2=k2;
  uint64_t r=splitmix64(k2);
  st->r0=(int)(( r       &63u)|1u);
  st->r1=(int)(((r>> 8u) &63u)|1u);
  st->r2=(int)(((r>>16u) &63u)|1u);
}
uint64_t mmc_next_u64(mmc_state_t* st){
  st->s += st->gamma;
  uint64_t y = mmc_permute64(st->s, st);
  return pcg_dxsm(y);
}
double mmc_next01(mmc_state_t* st){
  uint64_t y = mmc_next_u64(st);
  uint64_t mant = (y >> 11) & ((UINT64_C(1) << 53) - 1);
  return (double)mant / (double)(UINT64_C(1) << 53);
}
