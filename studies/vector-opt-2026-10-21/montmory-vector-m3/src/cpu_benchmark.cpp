#include "worker_pool.hpp"
#include <memory>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <thread>
#include <vector>
using montmory_stream::Engine;
static const char* names[]={"scalar_indexed_range","scalar_pair","simd4","simd8","pair_reused_workers"};
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static void generate(std::vector<uint64_t>& out,uint64_t seed,uint64_t base,unsigned e,unsigned workers) {
    if(workers==1){montmory_stream::fill(out.data(),out.size(),seed,base,Engine(e));return;}
    std::vector<std::thread> threads;
    for(unsigned w=0;w<workers;++w)threads.emplace_back([&,w]{
        size_t a=out.size()*w/workers,b=out.size()*(w+1)/workers;
        montmory_stream::fill(out.data()+a,b-a,seed,base+a,Engine(e));
    });
    for(auto& t:threads)t.join();
}
int main(int argc,char** argv) {
 try {
    if(argc!=2)throw std::runtime_error("usage: cpu_benchmark results_directory");
    std::ofstream csv(std::string(argv[1])+"/cpu-measurements.csv");
    csv<<"trial,seed,base,engine,workers,words,seconds,ns_per_word,full_comparison_pass\n";
    const unsigned cpus=std::max(1u,std::thread::hardware_concurrency());
    std::vector<std::pair<unsigned,unsigned>> jobs;
    for(unsigned e=0;e<5;++e)for(unsigned w:{1u,2u,4u,8u})if(w<=cpus)jobs.emplace_back(e,w);
    std::map<unsigned,std::unique_ptr<montmory_stream::WorkerPool>> pools;
    for(unsigned w:{1u,2u,4u,8u})if(w<=cpus)pools[w]=std::make_unique<montmory_stream::WorkerPool>(w);
    std::ofstream summary(std::string(argv[1])+"/cpu-summary.csv");
    summary<<"words,engine,workers,median_ns_per_word,GB_per_second,median_paired_ratio_vs_pair,seeds_faster\n";
    uint64_t checked=0;
    for(size_t n:{size_t(65536),size_t(8388608)}) {
    std::map<std::pair<unsigned,unsigned>,std::vector<double>> measurements;
    std::mt19937 order(20261024);std::vector<uint64_t> expected(n),out(n);
    for(unsigned trial=0;trial<9;++trial) {
        uint64_t seed=splitmix64(3000+trial),base=(trial%2)*montmory_stream::cycle;
        // Original independent scalar C state, lane by lane.
        for(uint64_t lane=0;lane<n/4096;++lane) {
            mmc_state_t s;mmc_init(&s,seed,lane);s.s+=(base+4096*lane)*s.gamma;
            for(uint64_t j=0;j<4096;++j)expected[4096*lane+j]=mmc_next_u64(&s);
        }
        std::shuffle(jobs.begin(),jobs.end(),order);
        for(auto job:jobs) {
            for(size_t i=0;i<n;++i)out[i]=~expected[i];
            if(job.first==4)pools.at(job.second)->generate(out.data(),out.size(),seed,base);
            else generate(out,seed,base,job.first,job.second);
            if(out!=expected)throw std::runtime_error("CPU warmup mismatch");
            checked+=n;
            for(size_t i=0;i<n;++i)out[i]=~expected[i];
            auto start=std::chrono::steady_clock::now();if(job.first==4)pools.at(job.second)->generate(out.data(),out.size(),seed,base);
            else generate(out,seed,base,job.first,job.second);
            double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(out!=expected)throw std::runtime_error("CPU timed mismatch");
            checked+=n;
            double ns=seconds*1e9/n;measurements[job].push_back(ns);
            csv<<trial<<','<<seed<<','<<base<<','<<names[job.first]<<','<<job.second<<','<<n<<','<<std::setprecision(15)<<seconds<<','<<ns<<",true\n";
        }
        csv.flush();if(!csv)throw std::runtime_error("CPU CSV failure");
        std::cout<<"CPU "<<n<<" mots, "<<trial+1<<"/9 : toutes les sorties identiques."<<std::endl;
    }

    std::cout<<"\nCPU raw64 — temps appelant ; pair_reused_workers exclut seulement la construction initiale du pool\nmoteur                    workers     GB/s   ratio/pair   seeds\n";
    for(const auto& entry:measurements) {
        auto j=entry.first;double ns=median(entry.second);std::vector<double> ratios;unsigned wins=0;
        const auto& ref=measurements.at({1,j.second});
        for(unsigned i=0;i<9;++i){ratios.push_back(ref[i]/entry.second[i]);wins+=entry.second[i]<ref[i];}
        double ratio=median(ratios);
        summary<<n<<','<<names[j.first]<<','<<j.second<<','<<std::setprecision(15)<<ns<<','<<8/ns<<','<<ratio<<','<<wins<<'\n';
        std::cout<<std::left<<std::setw(27)<<names[j.first]<<std::right<<std::setw(5)<<j.second<<std::fixed<<std::setprecision(3)<<std::setw(10)<<8/ns<<std::setw(13)<<ratio<<"   "<<wins<<"/9\n";
    }
    }
    std::ofstream proof(std::string(argv[1])+"/cpu-conformance.json");
    proof<<"{\"status\":\"pass\",\"checked_words\":"<<checked<<",\"hardware_threads\":"<<cpus<<",\"trials\":9}\n";
    summary.flush();proof.flush();if(!summary||!proof)throw std::runtime_error("CPU report failure");
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
