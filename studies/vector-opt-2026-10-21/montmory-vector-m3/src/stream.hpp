#ifndef MONTMORY_VECTOR_STREAM_HPP
#define MONTMORY_VECTOR_STREAM_HPP
#include <algorithm>
#include <stdexcept>
#include "mmc_core.c"
#include "wide_cpu.hpp"
namespace montmory_stream {
// Versioned logical geometry. Hardware worker count never enters this contract.
constexpr uint64_t lanes=2048,draws=4096,cycle=lanes*draws;
enum class Engine {indexed,pair,simd4,simd8};
inline uint64_t lane_at(uint64_t global) {return (global/draws)%lanes;}
inline uint64_t word(uint64_t seed,uint64_t global) {
    mmc_state_t s;mmc_init(&s,seed,lane_at(global));
    s.s+=(global)*s.gamma;return mmc_next_u64(&s);
}
// Count must not wrap the GLOBAL index. The internal unsigned counter may wrap.
inline void fill(uint64_t* out,uint64_t count,uint64_t seed,uint64_t base,Engine engine,unsigned shift=0) {
    if((count && !out)||(count && count-1>UINT64_MAX-base)||(shift!=0 && shift!=11) || unsigned(engine)>unsigned(Engine::simd8))
        throw std::invalid_argument("Invalid stream range or format");
    mmc_state_t s;mmc_init(&s,seed,0);
    using namespace montmory_wide;
    Wide64 k0=key(s.k0),k1=key(s.k1),k2=key(s.k2);
    uint64_t done=0;
    while(done<count) {
        const uint64_t global=base+done;
        const uint64_t n=std::min(count-done,draws-global%draws);
        const uint64_t gamma=UINT64_C(0x9E3779B97F4A7C15)+2*lane_at(global);
        uint64_t counter=seed+(global+1)*gamma,j=0;
        if(engine==Engine::simd4 || engine==Engine::simd8) {
            Wide64 c0=counters(counter,gamma),c1=counters(counter+4*gamma,gamma);
            const unsigned batch=engine==Engine::simd8?8:4;
            Wide64 step=key(batch*gamma);
            for(;j+batch<=n;j+=batch) {
                Wide64 a=wide_round(c0,k0,s.r0,s.r1),b{};
                if(batch==8)b=wide_round(c1,k0,s.r0,s.r1);
                a=wide_round(a,k1,s.r1,s.r2);
                if(batch==8)b=wide_round(b,k1,s.r1,s.r2);
                a=wide_round(a,k2,s.r2,s.r0);
                if(batch==8)b=wide_round(b,k2,s.r2,s.r0);
                store(out+done+j,wide_mix(a),shift);
                if(batch==8)store(out+done+j+4,wide_mix(b),shift);
                c0=wide_add(c0,step);if(batch==8)c1=wide_add(c1,step);
            }
            counter+=j*gamma;
        } else if(engine==Engine::pair) {
            for(;j+2<=n;j+=2) {
                uint64_t a=perm3(counter,&s),b=perm3(counter+gamma,&s);
                out[done+j]=pcg_dxsm(a)>>shift;out[done+j+1]=pcg_dxsm(b)>>shift;
                counter+=2*gamma;
            }
        }
        for(;j<n;++j) {
            uint64_t c=engine==Engine::indexed?seed+(global+j+1)*gamma:counter;
            out[done+j]=pcg_dxsm(perm3(c,&s))>>shift;counter+=gamma;
        }
        done+=n;
    }
}
}
#endif
