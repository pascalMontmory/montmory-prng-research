#ifndef MONTMORY_BENCH_REFERENCE_HPP
#define MONTMORY_BENCH_REFERENCE_HPP
#include <Random123/threefry.h>
#include <Random123/philox.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "bench_params.h"
#include "mmc_core.c"

static_assert(sizeof(BenchParams)==88 && offsetof(BenchParams,physical_threads)==76,
              "CPU/Metal parameter layout");
static const char* const engineNames[]={"montmory_indexed","threefry2x32_20",
                                       "philox4x32_10","montmory_historical",
                                       "montmory_u32_indexed","montmory_u64_tiled",
                                       "montmory_u32_tiled","write_baseline_NOT_PRNG",
                                       "montmory_pair","montmory_quad","montmory_mulhi_pair",
                                       "montmory_spec_pair","montmory_spec_dual"};
static const char* const kernelNames[]={"montmory_indexed","threefry_bulk","philox_bulk","mmc_fill",
    "montmory_u32_indexed","montmory_u64_tiled","montmory_u32_tiled","write_baseline",
    "montmory_pair","montmory_quad","montmory_mulhi_pair","montmory_spec_pair","montmory_spec_dual"};
static constexpr unsigned engineCount=13;
static inline bool isMontmory(unsigned engine) { return engine==0 || (engine>=3 && engine<=6) || engine>=8; }
static inline bool isSpecialized(unsigned engine) { return engine==11 || engine==12; }
static inline bool hasReduction(unsigned engine) { return engine==1 || engine==2 || engine>=6; }
static inline bool powerOfTwo(uint64_t x) { return x && !(x&(x-1)); }

static inline BenchParams parameters(uint64_t seed,uint64_t base,uint64_t words,
                                     uint32_t lanes,uint32_t workers,uint32_t shift) {
    if(!powerOfTwo(words) || !powerOfTwo(lanes) || words<lanes || !workers ||
       words>(1u<<25) || workers>(1u<<20) ||
       (shift!=0 && shift!=11)) throw std::runtime_error("Invalid benchmark geometry");
    mmc_state_t st; mmc_init(&st,seed,0);
    BenchParams p{};
    p.seed=seed; p.base=base; p.words=words;
    p.k0=st.k0; p.k1=st.k1; p.k2=st.k2;
    p.r0=st.r0; p.r1=st.r1; p.r2=st.r2;
    uint64_t k=splitmix64(seed); p.key0=uint32_t(k);
    k=splitmix64(k); p.key1=uint32_t(k);
    p.logical_lanes=lanes; p.physical_threads=workers; p.output_shift=shift;
    p.reserved=UINT32_MAX;
    if(workers>=lanes && workers%lanes==0 && powerOfTwo(workers/lanes)) {
        unsigned width=workers/lanes; p.reserved=0;
        while(width>1) { ++p.reserved; width>>=1; }
    }
    uint64_t draws=words/lanes;
    while(draws>1) { ++p.draws_shift; draws>>=1; }
    return p;
}

static inline uint64_t referenceWord(unsigned engine,const BenchParams& p,uint64_t i) {
    uint64_t word;
    if(isMontmory(engine)) {
        uint64_t lane=(i>>p.draws_shift)&(p.logical_lanes-1);
        mmc_state_t st; mmc_init(&st,p.seed,lane);
        st.s+=(p.base+i)*st.gamma;
        word=mmc_next_u64(&st);
    } else if(engine==1) {
        uint64_t index=p.base+i;
        threefry2x32_ctr_t c={{uint32_t(index),uint32_t(index>>32)}};
        threefry2x32_key_t k={{p.key0,p.key1}};
        auto r=threefry2x32_R(20,c,k);
        word=uint64_t(r.v[0])|(uint64_t(r.v[1])<<32);
    } else if(engine==2) {
        uint64_t index=p.base+i/2;
        philox4x32_ctr_t c={{uint32_t(index),uint32_t(index>>32),0,0}};
        philox4x32_key_t k={{p.key0,p.key1}};
        auto r=philox4x32_R(10,c,k);
        unsigned j=unsigned(i%2)*2;
        word=uint64_t(r.v[j])|(uint64_t(r.v[j+1])<<32);
    } else {
        word=(p.base+i)^p.seed;
    }
    return word>>p.output_shift;
}

// Independent ownership formula for diagnostic reference sums. Bulk output
// referenceWord remains the unchanged original C generator.
static inline uint32_t reductionOwner(unsigned engine,const BenchParams& p,uint64_t i) {
    if(engine==2) return uint32_t((i/2)%p.physical_threads);
    if((engine==6 || engine>=8) && p.reserved!=UINT32_MAX) {
        uint64_t draws=UINT64_C(1)<<p.draws_shift;
        uint32_t width=1u<<p.reserved;
        unsigned batch=engine==9?4:(engine>=8?2:1);
        return uint32_t(i/draws)*width+uint32_t(((i%draws)/batch)%width);
    }
    return uint32_t(i%p.physical_threads);
}
static inline std::vector<uint64_t> referenceReduction(unsigned engine,const BenchParams& p) {
    std::vector<uint64_t> result(p.physical_threads,0);
    for(uint64_t i=0;i<p.words;++i) result[reductionOwner(engine,p,i)]^=referenceWord(engine,p,i);
    return result;
}

struct KnownAnswer { std::array<uint32_t,6> input{}; std::array<uint32_t,4> expected{}; };
static inline std::vector<KnownAnswer> knownAnswers(const char* filename) {
    std::vector<KnownAnswer> all;
    for(unsigned engine=1;engine<=2;++engine) {
        std::ifstream file(filename);
        if(!file) throw std::runtime_error("Missing official known-answer file");
        std::string line; unsigned found=0;
        while(std::getline(file,line)) {
            std::istringstream row(line); std::string family; int rounds;
            if(!(row>>family>>rounds)) continue;
            if(family!=(engine==1?"threefry2x32":"philox4x32") || rounds!=(engine==1?20:10)) continue;
            KnownAnswer a; unsigned count=engine==1?2:4; row>>std::hex;
            for(unsigned j=0;j<count;++j) if(!(row>>a.input[j])) throw std::runtime_error("Bad KAT counter");
            for(unsigned j=4;j<6;++j) if(!(row>>a.input[j])) throw std::runtime_error("Bad KAT key");
            for(unsigned j=0;j<count;++j) if(!(row>>a.expected[j])) throw std::runtime_error("Bad KAT answer");
            if(engine==1) {
                threefry2x32_ctr_t c={{a.input[0],a.input[1]}};
                threefry2x32_key_t k={{a.input[4],a.input[5]}};
                auto r=threefry2x32_R(20,c,k);
                for(unsigned j=0;j<2;++j) if(r.v[j]!=a.expected[j]) throw std::runtime_error("CPU Threefry KAT failed");
            } else {
                philox4x32_ctr_t c={{a.input[0],a.input[1],a.input[2],a.input[3]}};
                philox4x32_key_t k={{a.input[4],a.input[5]}};
                auto r=philox4x32_R(10,c,k);
                for(unsigned j=0;j<4;++j) if(r.v[j]!=a.expected[j]) throw std::runtime_error("CPU Philox KAT failed");
            }
            all.push_back(a); ++found;
        }
        if(found!=3) throw std::runtime_error("Expected three official KATs per engine");
    }
    return all;
}
#endif
