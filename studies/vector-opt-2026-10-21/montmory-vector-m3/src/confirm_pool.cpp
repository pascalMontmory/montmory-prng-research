#include "worker_pool.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
using namespace montmory_stream;
int main(int argc,char** argv) {
 try {
    if(argc!=2)throw std::runtime_error("usage: confirm_pool results_directory");
    constexpr size_t n=65536;
    WorkerPool pool(4);std::mt19937 order(20261025);uint64_t checked=0;
    std::ofstream csv(std::string(argv[1])+"/pool-confirmation.csv");
    csv<<"trial,round,repeat,seed,engine,words,seconds,full_comparison_pass\n";
    const char* names[]={"sequential_pair","spawn4_pair","pool4_pair"};
    std::vector<uint64_t> expected(n),out(n);
    auto run=[&](unsigned engine,uint64_t seed,uint64_t base) {
        if(engine==0)fill(out.data(),n,seed,base,Engine::pair);
        else if(engine==2)pool.generate(out.data(),n,seed,base);
        else {
            std::vector<std::thread> threads;
            for(unsigned w=0;w<4;++w)threads.emplace_back([&,w]{fill(out.data()+n*w/4,n/4,seed,base+n*w/4,Engine::pair);});
            for(auto& t:threads)t.join();
        }
    };
    for(unsigned trial=0;trial<7;++trial) {
        uint64_t seed=splitmix64(6000+trial),base=trial*cycle+4093;
        for(size_t i=0;i<n;++i)expected[i]=word(seed,base+i);
        for(unsigned round=0;round<3;++round) {
            std::vector<unsigned> jobs={0,1,2};std::shuffle(jobs.begin(),jobs.end(),order);
            for(unsigned e:jobs) {
                run(e,seed,base);if(out!=expected)throw std::runtime_error("Pool warm mismatch");
                for(unsigned r=0;r<5;++r) {
                    for(size_t i=0;i<n;++i)out[i]=~expected[i];
                    auto start=std::chrono::steady_clock::now();run(e,seed,base);
                    double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                    if(out!=expected)throw std::runtime_error("Pool confirmation mismatch");
                    checked+=n;
                    csv<<trial<<','<<round<<','<<r<<','<<seed<<','<<names[e]<<','<<n<<','<<std::setprecision(15)<<t<<",true\n";
                }
            }
        }
        csv.flush();if(!csv)throw std::runtime_error("Pool CSV failure");
    }
    std::cout<<"Pool confirmation PASS: "<<checked<<" mots chronometres compares.\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
