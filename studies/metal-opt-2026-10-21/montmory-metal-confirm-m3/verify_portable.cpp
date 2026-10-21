#include "src/confirm_statistics.hpp"
#include "src/confirmation_config.hpp"
#include "src/reference.hpp"
#include <cassert>
#include <iostream>
#include <set>

int main() {
    assert(confirmMedian({9,1,3})==3);
    assert(confirmMedian({9,1,3,7})==5);
    // Equal-size groups can have a pooled median different from the median of
    // within-group medians; the benchmark must preserve the seed grouping.
    std::vector<std::vector<double>> seedSamples={{0,0,100},{1,2,3},{4,4,4}};
    auto medians=confirmSeedMedians(seedSamples,3,3);
    assert(confirmMedian(medians)==2);
    assert(confirmMedian({0,0,100,1,2,3,4,4,4})==3);
    bool refused=false;
    try {confirmSeedMedians(seedSamples,3,15);} catch(const std::runtime_error&) {refused=true;}
    assert(refused);
    std::set<std::pair<unsigned,unsigned>> families;
    for(auto j:confirmJobs) {
        assert(j.engine<engineCount && j.workers>=j.group && j.workers%j.group==0);
        assert(j.shift==0 || j.shift==11);
        assert(families.emplace(j.engine,j.shift).second);
        for(unsigned trial=0;trial<7;++trial) {
            auto p=parameters(splitmix64(2000+trial),0,8388608,2048,j.workers,j.shift);
            assert(p.draws_shift==12 && p.logical_lanes==2048 && p.physical_threads==j.workers);
        }
    }
    assert(families.size()==22 && families.count({10,0}) && families.count({10,11}));
    std::set<uint64_t> oldSeeds={0,1,42,UINT64_MAX};
    for(unsigned i=0;i<9;++i)oldSeeds.insert(splitmix64(i));
    for(unsigned i=0;i<5;++i)oldSeeds.insert(splitmix64(1000+i));
    std::set<uint64_t> newSeeds;
    for(unsigned i=0;i<7;++i) {
        auto seed=splitmix64(2000+i);
        assert(!oldSeeds.count(seed) && newSeeds.insert(seed).second);
    }
    assert(knownAnswers("reference/kat_vectors").size()==6);
    std::cout<<"PASS: grouped statistics, 22 fixed configurations, 7 distinct fresh seeds, 6 official CPU KATs.\n";
}
