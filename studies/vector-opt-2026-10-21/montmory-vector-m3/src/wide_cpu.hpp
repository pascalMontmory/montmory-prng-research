#ifndef MONTMORY_WIDE_CPU_HPP
#define MONTMORY_WIDE_CPU_HPP
#include <cstdint>
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif
namespace montmory_wide {
typedef uint32_t V __attribute__((vector_size(16)));
inline V splat(uint32_t x) {return V{x,x,x,x};}
inline V mask_lt(V a,V b) {return __builtin_convertvector(a<b,V);}
inline V mask_ne(V a,V b) {return __builtin_convertvector(a!=b,V);}
inline V high_product(V a,V b) {
#if defined(__aarch64__)
    uint32x4_t x,y;std::memcpy(&x,&a,16);std::memcpy(&y,&b,16);
    uint64x2_t p=vmull_u32(vget_low_u32(x),vget_low_u32(y));
    uint64x2_t q=vmull_high_u32(x,y);
    uint32x4_t z=vcombine_u32(vshrn_n_u64(p,32),vshrn_n_u64(q,32));
    V result;std::memcpy(&result,&z,16);return result;
#elif defined(__SSE2__)
    __m128i x,y;std::memcpy(&x,&a,16);std::memcpy(&y,&b,16);
    __m128i even=_mm_mul_epu32(x,y);
    __m128i odd=_mm_mul_epu32(_mm_srli_epi64(x,32),_mm_srli_epi64(y,32));
    __m128i z=_mm_or_si128(_mm_srli_epi64(even,32),_mm_slli_epi64(_mm_srli_epi64(odd,32),32));
    V result;std::memcpy(&result,&z,16);return result;
#else
    return V{uint32_t(uint64_t(a[0])*b[0]>>32),uint32_t(uint64_t(a[1])*b[1]>>32),
             uint32_t(uint64_t(a[2])*b[2]>>32),uint32_t(uint64_t(a[3])*b[3]>>32)};
#endif
}
#include "wide_math.inc"
inline Wide64 key(uint64_t k) {return {splat(uint32_t(k)),splat(uint32_t(k>>32))};}
inline Wide64 counters(uint64_t c,uint64_t step) {
    uint64_t a=c,b=c+step,d=c+2*step,e=c+3*step;
    return {V{uint32_t(a),uint32_t(b),uint32_t(d),uint32_t(e)},
            V{uint32_t(a>>32),uint32_t(b>>32),uint32_t(d>>32),uint32_t(e>>32)}};
}
inline void store(uint64_t* out,Wide64 w,unsigned shift) {
    for(unsigned j=0;j<4;++j)out[j]=(uint64_t(w.lo[j])|(uint64_t(w.hi[j])<<32))>>shift;
}
}
#endif
