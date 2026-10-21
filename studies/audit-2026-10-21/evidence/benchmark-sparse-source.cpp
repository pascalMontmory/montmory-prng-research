// CPU comparison using the frozen Montmory core and official Random123 headers.
// All words returned by each Random123 block are consumed. No reduced rounds.
#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <omp.h>
#include "mmc_core.c"

template<class G> struct R123Engine {
    using C = typename G::ctr_type;
    using K = typename G::key_type;
    static constexpr size_t CBITS = sizeof(C{}.v[0])*8;
    static constexpr size_t CWORDS = sizeof(C{}.v)/sizeof(C{}.v[0]);
    static constexpr size_t KWORDS = sizeof(K{}.v)/sizeof(K{}.v[0]);
    static constexpr size_t WIDTH = CWORDS*CBITS/64;
    K key{};
    explicit R123Engine(uint64_t seed) {
        // Expand all key words outside the timed region. Avoid specializing
        // Threefry to keys with several fixed zero words in this comparison.
        for(auto& word:key.v) {
            seed=splitmix64(seed);
            word=static_cast<typename std::remove_reference<decltype(word)>::type>(seed);
        }
    }
    inline std::array<uint64_t,WIDTH> block(uint64_t index) const {
        C counter{};
        counter.v[0]=static_cast<decltype(C{}.v[0])>(index);
        if constexpr(CBITS==32) counter.v[1]=uint32_t(index>>32);
        const C result=G{}(counter,key);
        std::array<uint64_t,WIDTH> words{};
        for(size_t i=0;i<WIDTH;++i) {
            if constexpr(CBITS==32) words[i]=uint64_t(result.v[2*i])|(uint64_t(result.v[2*i+1])<<32);
            else words[i]=result.v[i];
        }
        return words;
    }
};

static inline uint32_t f_branchless(uint64_t x,uint64_t k,unsigned r0,unsigned r1) {
    uint32_t odd=uint32_t(rotl64(x*3+k,r0)^(x>>1));
    uint32_t even=uint32_t(rotl64(x+k,r1)^(x*5));
    uint32_t mask=uint32_t(0)-uint32_t(x&1);
    return even^((even^odd)&mask);
}
static inline uint64_t perm_branchless(uint64_t x,const mmc_state_t& s) {
    uint32_t L=uint32_t(x>>32),R=uint32_t(x),F;
    F=f_branchless(R,s.k0,s.r0,s.r1); uint32_t t=R; R=L^F; L=t;
    F=f_branchless(R,s.k1,s.r1,s.r2); t=R; R=L^F; L=t;
    F=f_branchless(R,s.k2,s.r2,s.r0); t=R; R=L^F; L=t;
    return (uint64_t(L)<<32)|R;
}
template<bool Branchless> struct Montmory {
    static constexpr size_t WIDTH=1;
    mmc_state_t state;
    explicit Montmory(uint64_t seed) { mmc_init(&state,seed,0); }
    inline std::array<uint64_t,1> block(uint64_t index) const {
        uint64_t x=state.s+(index+1)*state.gamma;
        if constexpr(Branchless) return {pcg_dxsm(perm_branchless(x,state))};
        return {pcg_dxsm(perm3(x,&state))};
    }
};

template<class G> size_t verify_kat(const std::string& family,int rounds,const char* filename) {
    std::ifstream input(filename);
    if(!input) throw std::runtime_error("Missing known-answer vectors");
    std::string line; size_t tested=0;
    while(std::getline(input,line)) {
        std::istringstream row(line); std::string name; int nr;
        if(!(row>>name>>nr) || name!=family || nr!=rounds) continue;
        typename G::ctr_type counter{},expected{};
        typename G::key_type key{};
        row>>std::hex;
        for(auto& x:counter.v) if(!(row>>x)) throw std::runtime_error("Invalid counter");
        for(auto& x:key.v) if(!(row>>x)) throw std::runtime_error("Invalid key");
        for(auto& x:expected.v) if(!(row>>x)) throw std::runtime_error("Invalid answer");
        auto actual=G{}(counter,key);
        for(size_t i=0;i<sizeof(actual.v)/sizeof(actual.v[0]);++i)
            if(actual.v[i]!=expected.v[i]) throw std::runtime_error("KAT mismatch: "+family);
        ++tested;
    }
    if(tested!=3) throw std::runtime_error("Expected three official vectors: "+family);
    return tested;
}

volatile uint64_t checksum_sink=0;
struct Measurement { double seconds; uint64_t checksum; int observed_threads; };
template<class Engine,bool Store,bool Sparse=false>
__attribute__((noinline)) Measurement run(uint64_t n,uint64_t seed,int workers,uint64_t* output,const uint64_t* indices) {
    Engine engine(seed);
    uint64_t checksum=0;
    int observed=0;
    const uint64_t blocks=Sparse?n:n/Engine::WIDTH;
    const auto start=std::chrono::steady_clock::now();
    #pragma omp parallel num_threads(workers) reduction(+:checksum) shared(observed)
    {
        #pragma omp single
        observed=omp_get_num_threads();
        #pragma omp for schedule(static)
        for(uint64_t i=0;i<blocks;++i) {
            if constexpr(Sparse) {
                uint64_t index=indices[i];
                auto words=engine.block(index/Engine::WIDTH);
                uint64_t word=words[index%Engine::WIDTH];
                if constexpr(Store) output[i]=word;
                else checksum+=word;
            } else {
                auto words=engine.block(i);
                for(size_t j=0;j<Engine::WIDTH;++j) {
                    if constexpr(Store) output[i*Engine::WIDTH+j]=words[j];
                    else checksum+=words[j];
                }
            }
        }
    }
    const auto stop=std::chrono::steady_clock::now();
    // Consume stored results outside the timed region, preventing dead stores.
    if constexpr(Store) for(uint64_t i=0;i<n;++i) checksum+=output[i];
    checksum_sink=checksum;
    return {std::chrono::duration<double>(stop-start).count(),checksum,observed};
}
using Run = Measurement(*)(uint64_t,uint64_t,int,uint64_t*,const uint64_t*);
struct Entry { const char* name; Run compute; Run store; };
template<class E> Entry entry(const char* name,bool sparse=false) {
    if(sparse) return {name,run<E,false,true>,run<E,true,true>};
    return {name,run<E,false,false>,run<E,true,false>};
}

int main(int argc,char** argv) {
    uint64_t n=argc>1?std::stoull(argv[1]):(1ULL<<23);
    int repetitions=argc>2?std::stoi(argv[2]):9;
    bool sparse=argc>3 && std::string(argv[3])=="sparse";
    if(n<4 || n%4 || repetitions<1) return 2;
    size_t kats=0;
    const char* kat="deps/random123/tests/kat_vectors";
    kats+=verify_kat<r123::Philox2x32>("philox2x32",10,kat);
    kats+=verify_kat<r123::Philox4x32>("philox4x32",10,kat);
    kats+=verify_kat<r123::Philox2x64>("philox2x64",10,kat);
    kats+=verify_kat<r123::Philox4x64>("philox4x64",10,kat);
    kats+=verify_kat<r123::Threefry2x32>("threefry2x32",20,kat);
    kats+=verify_kat<r123::Threefry4x32>("threefry4x32",20,kat);
    kats+=verify_kat<r123::Threefry2x64>("threefry2x64",20,kat);
    kats+=verify_kat<r123::Threefry4x64>("threefry4x64",20,kat);
    size_t equivalent=0;
    uint64_t seed=42;
    for(unsigned a=0;a<64;++a) {
        Montmory<false> original(seed); Montmory<true> alternative(seed);
        mmc_state_t serial; mmc_init(&serial,seed,0);
        for(uint64_t i=0;i<4096;++i) {
            auto x=original.block(i)[0];
            if(x!=alternative.block(i)[0] || x!=mmc_next_u64(&serial)) return 1;
            ++equivalent;
        }
        for(uint64_t i:{UINT64_MAX,UINT64_MAX-1,UINT64_C(0x8000000000000000)}) {
            if(original.block(i)!=alternative.block(i)) return 1;
            ++equivalent;
        }
        seed=splitmix64(seed);
    }
    std::cerr<<"KATs="<<kats<<" Montmory-equivalence="<<equivalent<<" sizeof-state="<<sizeof(mmc_state_t)<<"\n";
    std::vector<Entry> engines={entry<Montmory<false>>("montmory_original",sparse),entry<Montmory<true>>("montmory_branchless",sparse),
        entry<R123Engine<r123::Philox2x32>>("philox2x32_10",sparse),entry<R123Engine<r123::Philox4x32>>("philox4x32_10",sparse),
        entry<R123Engine<r123::Philox2x64>>("philox2x64_10",sparse),entry<R123Engine<r123::Philox4x64>>("philox4x64_10",sparse),
        entry<R123Engine<r123::Threefry2x32>>("threefry2x32_20",sparse),entry<R123Engine<r123::Threefry4x32>>("threefry4x32_20",sparse),
        entry<R123Engine<r123::Threefry2x64>>("threefry2x64_20",sparse),entry<R123Engine<r123::Threefry4x64>>("threefry4x64_20",sparse)};
    std::vector<uint64_t> output(n);
    std::vector<uint64_t> indices(sparse?n:0);
    if(sparse) { uint64_t x=123; for(auto& i:indices) { x=splitmix64(x); i=x; } }
    std::mt19937 order(20261021);
    omp_set_dynamic(0);
    for(auto e:engines) e.store(std::min(n,uint64_t(1<<16)),42,1,output.data(),indices.data());
    std::cout<<"engine,mode,workers,observed_workers,trial,seed,words,seconds,ns_per_word,checksum\n";
    for(int trial=0;trial<repetitions;++trial) {
        uint64_t s=trial==0?42:splitmix64(uint64_t(trial));
        std::shuffle(engines.begin(),engines.end(),order);
        for(int workers:{1,4}) for(int mode=0;mode<2;++mode) for(auto e:engines) {
            auto m=(mode?e.store:e.compute)(n,s,workers,output.data(),indices.data());
            if(m.observed_threads!=workers) return 1;
            std::cout<<e.name<<','<<(mode?"store64":"compute")<<','<<workers<<','<<m.observed_threads<<','<<trial<<','<<s<<','<<n<<','
                     <<std::setprecision(10)<<m.seconds<<','<<m.seconds*1e9/n<<','<<std::hex<<m.checksum<<std::dec<<'\n';
        }
        std::cout.flush();
    }
}
