#include "reference.hpp"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>

struct Job {unsigned lanes,workers;bool geometry;};
static constexpr uint64_t N=8388608;
static std::vector<uint64_t> reference(uint64_t seed,unsigned lanes) {
    std::vector<uint64_t> out(N);const uint64_t draws=N/lanes;
    for(unsigned lane=0;lane<lanes;++lane) {
        mmc_state_t state;mmc_init(&state,seed,lane);
        state.s+=(uint64_t(lane)*draws)*state.gamma;
        for(uint64_t j=0;j<draws;++j) out[uint64_t(lane)*draws+j]=mmc_next_u64(&state);
    }
    return out;
}
static int generate(uint64_t seed,unsigned lanes,unsigned workers,std::vector<uint64_t>& out) {
    mmc_state_t st;mmc_init(&st,seed,0);unsigned shift=0;
    for(uint64_t d=N/lanes;d>1;d>>=1) ++shift;
    int actual=0;
    #pragma omp parallel num_threads(workers) shared(actual)
    {
        #pragma omp single
        actual=omp_get_num_threads();
        #pragma omp for schedule(static)
        for(uint64_t i=0;i<N;++i) {
            const uint64_t lane=(i>>shift)&(lanes-1);
            const uint64_t s=seed+(i+1)*(UINT64_C(0x9E3779B97F4A7C15)+2*lane);
            out[i]=pcg_dxsm(perm3(s,&st));
        }
    }
    return actual;
}
int main(int argc,char**argv) {
  try {
    if(argc!=2) throw std::runtime_error("usage: cpu_lanes result_directory");
    omp_set_dynamic(0);
    std::ofstream csv(std::string(argv[1])+"/cpu_measurements.csv");
    std::ofstream changes(std::string(argv[1])+"/cpu_geometry_changes.csv");
    if(!csv || !changes) throw std::runtime_error("Cannot open results");
    csv<<"phase,trial,seed,logical_lanes,draws_per_lane,requested_workers,actual_workers,words,seconds,ns_per_word,checksum\n";
    changes<<"trial,seed,logical_lanes,draws_per_lane,different_words_from_T2048_D4096\n";
    std::vector<Job> jobs;
    for(unsigned w:{1u,2u,4u,8u,16u}) jobs.push_back({2048,w,false});
    for(unsigned l:{256u,512u,1024u,4096u,8192u,16384u}) jobs.push_back({l,4,true});
    std::mt19937 order(20261021);std::vector<uint64_t> out(N);
    uint64_t comparisons=0;
    for(unsigned trial=0;trial<9;++trial) {
        const uint64_t seed=trial?splitmix64(trial):42;
        std::map<unsigned,std::vector<uint64_t>> references;
        for(unsigned l:{256u,512u,1024u,2048u,4096u,8192u,16384u}) references.emplace(l,reference(seed,l));
        const auto& baseline=references.at(2048);
        for(const auto& entry:references) {
            uint64_t differences=0;
            for(uint64_t i=0;i<N;++i) differences+=entry.second[i]!=baseline[i];
            changes<<trial<<','<<seed<<','<<entry.first<<','<<(N/entry.first)<<','<<differences<<'\n';
        }
        std::shuffle(jobs.begin(),jobs.end(),order);
        for(auto j:jobs) {
            generate(seed,j.lanes,j.workers,out); // Warm the current configuration.
            auto start=std::chrono::steady_clock::now();
            int actual=generate(seed,j.lanes,j.workers,out);
            double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(std::memcmp(out.data(),references.at(j.lanes).data(),N*sizeof(uint64_t)))
                throw std::runtime_error("Parallel output differs from original C reference");
            comparisons+=N;
            uint64_t checksum=0;for(auto value:out)checksum^=value;
            csv<<(j.geometry?"logical_geometry_changes_stream":"physical_workers_same_stream")<<','<<trial<<','<<seed<<','<<j.lanes<<','<<(N/j.lanes)<<','<<j.workers<<','<<actual<<','<<N<<','<<std::setprecision(12)<<seconds<<','<<seconds*1e9/N<<','<<checksum<<'\n';
        }
        csv.flush();changes.flush();
        std::cout<<"CPU repetition "<<trial+1<<"/9 terminee"<<std::endl;
    }
    if(!csv || !changes) throw std::runtime_error("Write error");
    std::ofstream proof(std::string(argv[1])+"/cpu_conformance.json");
    proof<<"{\"status\":\"pass\",\"fully_compared_words\":"<<comparisons<<",\"reference\":\"original C mmc_next_u64, lane by lane\",\"openmp_dynamic\":false}\n";
    std::cout<<"PASS: "<<comparisons<<" mots compares exactement."<<std::endl;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
