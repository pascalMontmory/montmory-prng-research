#define main cpu_benchmark_entry
#include "../src/cpu_benchmark.cpp"
#undef main
int main() {
    std::vector<uint64_t> out(N+16,UINT64_MAX);uint64_t checked=0;
    const std::pair<uint64_t,uint64_t> intervals[]={{0,0},{0,1},{1,7},{D-3,D+5},{D+1,2*D+3},{N-13,N},{N,N}};
    for(uint64_t seed:{UINT64_C(0),UINT64_C(1),UINT64_C(42),UINT64_MAX})for(unsigned e=0;e<4;++e) {
        auto p=parameters(seed,0,N,2048,65536,0);
        for(auto interval:intervals) {
            uint64_t a=interval.first,b=interval.second;
            if(a)out[a-1]=UINT64_MAX;
            out[b]=UINT64_MAX;
            for(uint64_t i=a;i<b;++i)out[i]=~referenceWord(10,p,i);
            range(out.data(),a,b,seed,e);
            for(uint64_t i=a;i<b;++i) {
                if(out[i]!=referenceWord(10,p,i))return 1;
                ++checked;
            }
            if((a && out[a-1]!=UINT64_MAX)||out[b]!=UINT64_MAX)return 2;
        }
    }
    std::cout<<"{\"status\":\"pass\",\"boundary_words\":"<<checked<<",\"guards\":\"pass\"}\n";
}
