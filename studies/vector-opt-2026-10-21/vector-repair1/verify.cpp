#include "worker_pool.hpp"
#include <array>
#include <iostream>
#include <random>
#include <vector>
#include <thread>
using montmory_stream::Engine;
static uint64_t checked=0;
static void require(bool ok) {if(!ok)throw std::runtime_error("Conformance mismatch");}
static void test(uint64_t seed,uint64_t base,size_t n) {
    std::vector<uint64_t> expected(n),out(n+2);
    for(size_t i=0;i<n;++i)expected[i]=montmory_stream::word(seed,base+i);
    for(auto e:{Engine::indexed,Engine::pair,Engine::simd4,Engine::simd8})for(unsigned shift:{0u,11u}) {
        for(size_t i=0;i<n;++i)out[i+1]=~(expected[i]>>shift);
        out[0]=out[n+1]=0x1a23456789abcdefULL;
        montmory_stream::fill(out.data()+1,n,seed,base,e,shift);
        for(size_t i=0;i<n;++i){require(out[i+1]==(expected[i]>>shift));++checked;}
        require(out[0]==0x1a23456789abcdefULL && out[n+1]==out[0]);
    }
}
int main() {
 try {
    using namespace montmory_wide;
    std::mt19937_64 random(0x6d6f6e746d6f7279ULL);
    uint64_t components=0;
    for(unsigned trial=0;trial<10000;++trial) {
        V a{},b{};for(unsigned j=0;j<4;++j){a[j]=uint32_t(random());b[j]=uint32_t(random());}
        V h=high_product(a,b);
        for(unsigned j=0;j<4;++j){require(h[j]==uint32_t(uint64_t(a[j])*b[j]>>32));++components;}
        uint64_t k=random();unsigned ro=2*(trial%32)+1,re=2*((trial/32)%32)+1;
        V f=wide_f(a,key(k),ro,re);
        for(unsigned j=0;j<4;++j){require(f[j]==mmc_f(a[j],k,ro,re));++components;}
        // Cover every final rotation (including zero) explicitly.
        Wide64 x={a,(b&splat(0x07ffffff))|splat((trial%32)<<27)};
        auto y=wide_mix(x);
        for(unsigned j=0;j<4;++j){require((uint64_t(y.lo[j])|(uint64_t(y.hi[j])<<32))==pcg_dxsm(uint64_t(x.lo[j])|(uint64_t(x.hi[j])<<32)));++components;}
    }
    for(uint64_t seed:{UINT64_C(0),UINT64_C(1),UINT64_C(42),UINT64_MAX}) {
        for(uint64_t base:{UINT64_C(0),UINT64_C(1),UINT64_C(4093),UINT64_C(4095),UINT64_C(8388605),UINT64_C(8388608),UINT64_C(16777213),UINT64_MAX-8192})
            for(size_t n:{0u,1u,3u,4u,7u,8u,9u,65u,4099u,8193u})test(seed,base,n);
        test(seed,UINT64_MAX,1);
    }
    // Sequence draws explicitly: function argument evaluation order is unspecified.
    for(unsigned i=0;i<128;++i) {
        const uint64_t test_seed=random();
        const uint64_t test_base=random()>>1;
        const size_t test_size=random()%8192;
        test(test_seed,test_base,test_size);
    }
    // Original historical geometry, including a refill; independent scalar C state.
    uint64_t historical=0;
    for(uint64_t seed:{UINT64_C(42),UINT64_C(0),UINT64_MAX}) {
        std::vector<uint64_t> out(montmory_stream::cycle);
        for(uint64_t base:{UINT64_C(0),montmory_stream::cycle}) {
            montmory_stream::fill(out.data(),out.size(),seed,base,Engine::simd4);
            for(uint64_t lane=0;lane<2048;++lane) {
                mmc_state_t s;mmc_init(&s,seed,lane);s.s+=(base+lane*4096)*s.gamma;
                for(uint64_t j=0;j<4096;++j){require(out[lane*4096+j]==mmc_next_u64(&s));++historical;}
            }
        }
    }
    // Arbitrary scheduling, odd chunks, boundaries and software worker counts.
    uint64_t partitioned=0;
    for(unsigned workers:{1u,2u,3u,7u,8u}) {
        const size_t n=131089;const uint64_t base=8388591;
        std::vector<uint64_t> out(n),expected(n);
        montmory_stream::fill(expected.data(),n,42,base,Engine::indexed);
        std::vector<std::thread> threads;
        for(unsigned w=0;w<workers;++w)threads.emplace_back([&,w]{
            size_t first=n*w/workers,end=n*(w+1)/workers;
            while(first<end){size_t chunk=std::min(end-first,size_t(137));
                montmory_stream::fill(out.data()+first,chunk,42,base+first,Engine::simd8);first+=chunk;}
        });
        for(auto& t:threads)t.join();
        require(out==expected);partitioned+=n;
    }
    uint64_t pooled=0;
    for(unsigned workers:{1u,2u,3u,7u,8u}) {
        montmory_stream::WorkerPool pool(workers);
        for(unsigned trial=0;trial<100;++trial) {
            uint64_t base=trial%2?8388605:UINT64_MAX-10000,seed=random();
            size_t n=trial<10?trial:random()%8192;
            std::vector<uint64_t> out(n),expected(n);
            for(size_t i=0;i<n;++i)expected[i]=montmory_stream::word(seed,base+i)>>(trial%2?11:0);
            pool.generate(out.data(),n,seed,base,Engine(trial%4),trial%2?11:0);
            require(out==expected);pooled+=n;
        }
    }
    for(unsigned bad=0;bad<3;++bad) {
        bool rejected=false;uint64_t out[2];
        try{montmory_stream::fill(bad==0?nullptr:out,2,42,bad==1?UINT64_MAX:0,Engine::simd4,bad==2?7:0);}
        catch(const std::invalid_argument&){rejected=true;}require(rejected);
    }
    std::cout<<"{\"status\":\"pass\",\"range_words\":"<<checked<<",\"component_checks\":"<<components
        <<",\"historical_words\":"<<historical<<",\"partitioned_words\":"<<partitioned<<",\"pooled_words\":"<<pooled<<"}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
