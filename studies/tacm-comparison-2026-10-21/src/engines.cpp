#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <array>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iostream>
#include "mmc_core.c"

namespace official_xoshiro {
#include "xoshiro_reference_prefix.h"
}

static inline uint32_t f_fast(uint32_t x,uint64_t k,unsigned a,unsigned b) {
    const uint64_t w=x;
    const uint32_t odd=uint32_t(rotl64(w*3+k,a)^(w>>1));
    const uint32_t even=uint32_t(rotl64(w+k,b)^(w*5));
    const uint32_t mask=uint32_t(0)-uint32_t(x&1);
    return even^((odd^even)&mask);
}
static inline uint64_t mont_output(uint64_t x,const mmc_state_t& s) {
    uint32_t l=uint32_t(x>>32),r=uint32_t(x),t;
    t=r; r=l^f_fast(r,s.k0,s.r0,s.r1); l=t;
    t=r; r=l^f_fast(r,s.k1,s.r1,s.r2); l=t;
    t=r; r=l^f_fast(r,s.k2,s.r2,s.r0); l=t;
    return pcg_dxsm((uint64_t(l)<<32)|r);
}

struct CounterEngine {
    int engine;
    uint64_t seed;
    mmc_state_t st;
    uint32_t k0,k1;
    r123::Threefry4x64::key_type k64{};
    CounterEngine(int e,uint64_t s):engine(e),seed(s) {
        mmc_init(&st,seed,0);
        uint64_t k=splitmix64(seed); k0=uint32_t(k);
        k=splitmix64(k); k1=uint32_t(k);
        k=seed;
        for(auto& word:k64.v) { k=splitmix64(k); word=k; }
    }
    inline uint64_t word(uint64_t i) const {
        if(engine==0) {
            const uint64_t lane=(i/4096)%2048;
            const uint64_t gamma=UINT64_C(0x9E3779B97F4A7C15)+2*lane;
            return mont_output(seed+(i+1)*gamma,st);
        }
        if(engine==1) {
            const uint64_t block=i/2;
            r123::Philox4x32::ctr_type c={{uint32_t(block),uint32_t(block>>32),0,0}};
            r123::Philox4x32::key_type k={{k0,k1}};
            const auto v=r123::Philox4x32{}(c,k);
            const unsigned j=unsigned(i%2)*2;
            return uint64_t(v.v[j])|(uint64_t(v.v[j+1])<<32);
        }
        if(engine==2) {
            r123::Threefry2x32::ctr_type c={{uint32_t(i),uint32_t(i>>32)}};
            r123::Threefry2x32::key_type k={{k0,k1}};
            const auto v=r123::Threefry2x32{}(c,k);
            return uint64_t(v.v[0])|(uint64_t(v.v[1])<<32);
        }
        r123::Threefry4x64::ctr_type c={{i/4,0,0,0}};
        return r123::Threefry4x64{}(c,k64).v[i%4];
    }
    void fill(uint64_t base,size_t n,uint64_t* out) const {
        size_t i=0;
        if(engine==1) {
            if(base%2 && n) { out[0]=word(base); ++i; }
            for(;i+1<n;i+=2) {
                const uint64_t b=(base+i)/2;
                r123::Philox4x32::ctr_type c={{uint32_t(b),uint32_t(b>>32),0,0}};
                r123::Philox4x32::key_type k={{k0,k1}};
                const auto v=r123::Philox4x32{}(c,k);
                out[i]=uint64_t(v.v[0])|(uint64_t(v.v[1])<<32);
                out[i+1]=uint64_t(v.v[2])|(uint64_t(v.v[3])<<32);
            }
        } else if(engine==3) {
            for(;i<n && (base+i)%4;++i) out[i]=word(base+i);
            for(;i+3<n;i+=4) {
                r123::Threefry4x64::ctr_type c={{(base+i)/4,0,0,0}};
                const auto v=r123::Threefry4x64{}(c,k64);
                for(unsigned j=0;j<4;++j) out[i+j]=v.v[j];
            }
        }
        for(;i<n;++i) out[i]=word(base+i);
    }
};

struct Xoshiro {
    uint64_t s[4];
    explicit Xoshiro(uint64_t seed) {
        for(auto& v:s) { v=splitmix64(seed); seed+=UINT64_C(0x9E3779B97F4A7C15); }
    }
    uint64_t next() {
        const uint64_t r=rotl64(s[1]*5,7)*9,t=s[1]<<17;
        s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3];
        s[2]^=t; s[3]=rotl64(s[3],45);
        return r;
    }
};

extern "C" int tacm_fill(int engine,uint64_t seed,uint64_t base,size_t n,uint64_t* out) {
    if(engine<0 || engine>4 || !out) return 1;
    if(engine<4) CounterEngine(engine,seed).fill(base,n,out);
    else {
        if(base) return 2;
        Xoshiro x(seed);
        for(size_t i=0;i<n;++i) out[i]=x.next();
    }
    return 0;
}
extern "C" int tacm_gather(int engine,uint64_t seed,size_t n,const uint64_t* indices,uint64_t* out) {
    if(engine<0 || engine>3 || !out || !indices) return 1;
    CounterEngine x(engine,seed);
    for(size_t i=0;i<n;++i) out[i]=x.word(indices[i]);
    return 0;
}

template<class G> int kats(const char* name,int rounds,const char* path) {
    std::ifstream file(path); std::string line; int count=0;
    while(std::getline(file,line)) {
        std::istringstream row(line); std::string family; int r;
        if(!(row>>family>>r) || family!=name || r!=rounds) continue;
        typename G::ctr_type c{},expected{}; typename G::key_type k{};
        row>>std::hex;
        for(auto& v:c.v) if(!(row>>v)) return -100;
        for(auto& v:k.v) if(!(row>>v)) return -100;
        for(auto& v:expected.v) if(!(row>>v)) return -100;
        auto actual=G{}(c,k);
        for(unsigned i=0;i<sizeof(c.v)/sizeof(c.v[0]);++i) if(actual.v[i]!=expected.v[i]) return -100;
        ++count;
    }
    return count;
}

extern "C" int tacm_check(const char* kat) {
    if(kats<r123::Philox4x32>("philox4x32",10,kat)!=3 ||
       kats<r123::Threefry2x32>("threefry2x32",20,kat)!=3 ||
       kats<r123::Threefry4x64>("threefry4x64",20,kat)!=3) return 1;
    uint64_t seed=0;
    for(unsigned s=0;s<16;++s) {
        if(s==1) seed=1; else if(s==2) seed=42; else if(s==3) seed=UINT64_MAX;
        else if(s>3) seed=splitmix64(seed);
        CounterEngine m(0,seed);
        Xoshiro x(seed);
        std::copy(x.s,x.s+4,official_xoshiro::s);
        for(uint64_t i=0;i<65536;++i) {
            mmc_state_t original;
            mmc_init(&original,seed,(i/4096)%2048);
            original.s+=i*original.gamma;
            if(m.word(i)!=mmc_next_u64(&original)) return 2;
            if(x.next()!=official_xoshiro::next()) return 3;
        }
        for(uint64_t i:{UINT64_MAX,UINT64_MAX-1,UINT64_C(1)<<63,UINT64_C(8388607),UINT64_C(8388608)}) {
            mmc_state_t original;
            mmc_init(&original,seed,(i/4096)%2048); original.s+=i*original.gamma;
            if(m.word(i)!=mmc_next_u64(&original)) return 4;
        }
        for(int e=0;e<4;++e) for(uint64_t base:{0ULL,1ULL,3ULL,4095ULL,8388607ULL}) {
            CounterEngine engine(e,seed); uint64_t words[257]; engine.fill(base,257,words);
            for(size_t i=0;i<257;++i) if(words[i]!=engine.word(base+i)) return 5;
        }
    }
    return 0;
}

#ifdef TACM_CHECK_MAIN
int main(int argc,char** argv) {
    if(argc!=2) return 99;
    int code=tacm_check(argv[1]);
    std::cout<<"{\"native_conformance_status\":"<<code
             <<",\"Random123_KATs\":9,\"Montmory_original_words\":1048656"
               ",\"xoshiro_official_words\":1048576,\"bulk_indexed_words\":82240}\n";
    return code;
}
#endif
