#include "lane_reference.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <thread>
#include <tuple>

static constexpr uint64_t N=8388608,D=4096;
static const char* const names[]={"indexed","incremental","incremental_pair","incremental_quad"};
template<unsigned Batch>
static void incremental(uint64_t* out,uint64_t begin,uint64_t end,uint64_t seed,const mmc_state_t& state) {
    while(begin<end) {
        uint64_t lane=begin/D,stop=std::min(end,(lane+1)*D);
        uint64_t gamma=UINT64_C(0x9E3779B97F4A7C15)+2*lane;
        uint64_t counter=seed+(begin+1)*gamma;
        for(;begin+Batch<=stop;begin+=Batch) {
            uint64_t x[Batch];
            for(unsigned b=0;b<Batch;++b)x[b]=perm3(counter+uint64_t(b)*gamma,&state);
            for(unsigned b=0;b<Batch;++b)out[begin+b]=pcg_dxsm(x[b]);
            counter+=Batch*gamma;
        }
        for(;begin<stop;++begin){out[begin]=pcg_dxsm(perm3(counter,&state));counter+=gamma;}
    }
}
static void range(uint64_t* out,uint64_t begin,uint64_t end,uint64_t seed,unsigned engine) {
    mmc_state_t state;mmc_init(&state,seed,0);
    if(engine==0) {
        for(uint64_t i=begin;i<end;++i) {
            uint64_t gamma=UINT64_C(0x9E3779B97F4A7C15)+2*(i/D);
            out[i]=pcg_dxsm(perm3(seed+(i+1)*gamma,&state));
        }
    } else if(engine==1)incremental<1>(out,begin,end,seed,state);
    else if(engine==2)incremental<2>(out,begin,end,seed,state);
    else incremental<4>(out,begin,end,seed,state);
}
static void generate(std::vector<uint64_t>& out,uint64_t seed,unsigned engine,unsigned workers) {
    if(workers==1){range(out.data(),0,N,seed,engine);return;}
    std::vector<std::thread> threads;threads.reserve(workers);
    for(unsigned w=0;w<workers;++w)threads.emplace_back(range,out.data(),N*w/workers,N*(w+1)/workers,seed,engine);
    for(auto& t:threads)t.join();
}
static double quantile(std::vector<double> values,double q) {
    std::sort(values.begin(),values.end());double index=(values.size()-1)*q;
    size_t lo=size_t(index),hi=std::min(lo+1,values.size()-1);
    return values[lo]+(index-lo)*(values[hi]-values[lo]);
}
int main(int argc,char**argv) {
 try {
    if(argc!=2)throw std::runtime_error("usage: cpu_benchmark result_directory");
    std::ofstream csv(std::string(argv[1])+"/cpu-measurements.csv");
    if(!csv)throw std::runtime_error("Cannot open CPU CSV");
    csv<<"trial,seed,engine,workers,words,seconds,ns_per_word,full_comparison_pass\n";
    unsigned cpus=std::max(1u,std::thread::hardware_concurrency());
    std::vector<std::pair<unsigned,unsigned>> jobs;
    for(unsigned e=0;e<4;++e)for(unsigned w:{1u,2u,4u,8u,16u})if(w<=cpus)jobs.emplace_back(e,w);
    std::map<std::pair<unsigned,unsigned>,std::vector<double>> results;
    std::mt19937 order(20261023);std::vector<uint64_t> out(N);uint64_t compared=0;
    for(unsigned trial=0;trial<9;++trial) {
        uint64_t seed=trial?splitmix64(trial):42;
        auto expected=laneReference(parameters(seed,0,N,2048,65536,0),10);
        std::shuffle(jobs.begin(),jobs.end(),order);
        for(auto j:jobs) {
            for(uint64_t i=0;i<N;++i)out[i]=~expected[i];
            generate(out,seed,j.first,j.second);
            if(std::memcmp(out.data(),expected.data(),N*8))throw std::runtime_error("CPU warm-up mismatch");
            compared+=N;
            auto start=std::chrono::steady_clock::now();
            generate(out,seed,j.first,j.second);
            double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(std::memcmp(out.data(),expected.data(),N*8))throw std::runtime_error("CPU timed output mismatch");
            compared+=N;double ns=seconds*1e9/N;results[j].push_back(ns);
            csv<<trial<<','<<seed<<','<<names[j.first]<<','<<j.second<<','<<N<<','<<std::setprecision(12)<<seconds<<','<<ns<<",true\n";
        }
        csv.flush();if(!csv)throw std::runtime_error("CPU CSV write failed");
        std::cout<<"CPU repetition "<<trial+1<<"/9 terminee, flux identique."<<std::endl;
    }
    std::ofstream summary(std::string(argv[1])+"/cpu-summary.csv");
    summary<<"engine,workers,median_ns_per_word,q10,q90,stored_GB_per_second,speedup_vs_indexed_same_workers\n";
    std::cout<<"\nCPU raw64 — T=2048,D=4096 ; creation/jonction des workers incluse\n"
             <<"moteur                workers    ns/mot       GB/s   ratio/indexe\n";
    for(const auto& entry:results) {
        unsigned e=entry.first.first,w=entry.first.second;double ns=quantile(entry.second,.5);
        double ratio=quantile(results.at({0,w}),.5)/ns;
        summary<<names[e]<<','<<w<<','<<std::setprecision(12)<<ns<<','<<quantile(entry.second,.1)<<','<<quantile(entry.second,.9)<<','<<8/ns<<','<<ratio<<'\n';
        std::cout<<std::left<<std::setw(23)<<names[e]<<std::right<<std::setw(6)<<w<<std::setw(11)<<std::fixed<<std::setprecision(3)<<ns<<std::setw(11)<<8/ns<<std::setw(13)<<ratio<<'\n';
    }
    std::ofstream report(std::string(argv[1])+"/cpu-conformance.json");
    report<<"{\"status\":\"pass\",\"fully_compared_words\":"<<compared<<",\"reported_hardware_threads\":"<<cpus
          <<",\"configurations\":"<<jobs.size()<<",\"trials\":9,\"scope\":\"Original scalar C oracle; std::thread workers, no affinity; raw64\"}\n";
    summary.flush();report.flush();if(!summary||!report)throw std::runtime_error("CPU report write failed");
    return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
