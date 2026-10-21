// Frozen after search: indexed 1/16 workers versus incremental_pair 1/8.
// Confirmation uses new seeds; no re-selection is performed here.
#define main cpu_benchmark_entry
#include "../src/cpu_benchmark.cpp"
#undef main
int main(int argc,char**argv) {
    if(argc!=2)return 2;
    std::ofstream csv(argv[1]);if(!csv)return 3;
    csv<<"trial,seed,engine,workers,ns_per_word,full_comparison_pass\n";
    std::vector<std::pair<unsigned,unsigned>> jobs={{0,1},{2,1},{0,16},{2,8}};
    std::vector<uint64_t> out(N);std::mt19937 order(20261024);
    for(unsigned trial=0;trial<5;++trial) {
        uint64_t seed=splitmix64(1000+trial);
        auto expected=laneReference(parameters(seed,0,N,2048,65536,0),10);
        std::shuffle(jobs.begin(),jobs.end(),order);
        for(auto j:jobs) {
            generate(out,seed,j.first,j.second);
            auto start=std::chrono::steady_clock::now();generate(out,seed,j.first,j.second);
            double ns=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()*1e9/N;
            if(std::memcmp(out.data(),expected.data(),N*8))return 1;
            csv<<trial<<','<<seed<<','<<names[j.first]<<','<<j.second<<','<<std::setprecision(12)<<ns<<",true\n";
        }
    }
    csv.flush();if(!csv)return 4;
    std::cout<<"CPU confirmation PASS: 20 outputs, 167772160 words compared\n";
}
