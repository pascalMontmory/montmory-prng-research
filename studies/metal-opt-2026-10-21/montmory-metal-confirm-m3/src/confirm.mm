#include "v3_helpers.hpp"
#include "lane_reference.hpp"
#include "confirmation_config.hpp"
#include "confirm_statistics.hpp"
#include <cmath>

using ConfirmFamily = std::pair<unsigned, unsigned>;
static NSDictionary* configJSON(ConfirmJob j) {
    return @{@"engine":[NSString stringWithUTF8String:engineNames[j.engine]],
        @"format":j.shift?@"mantissa53":@"raw64", @"threads":@(j.workers), @"group":@(j.group)};
}
static void compareBuffer(id<MTLBuffer> buffer, const std::vector<uint64_t>& expected, ConfirmJob j) {
    const uint64_t* actual = static_cast<const uint64_t*>(buffer.contents);
    for (size_t i=0; i<expected.size(); ++i)
        if (actual[i] != (expected[i] >> j.shift))
            throw std::runtime_error(std::string(engineNames[j.engine])+" mismatch at "+std::to_string(i));
    for (size_t i=expected.size(); i<expected.size()+16; ++i)
        require(actual[i]==UINT64_MAX,"Output guard overwritten");
}
int main(int argc, char** argv) {
 @autoreleasepool {
  try {
    require(argc==3,"usage: confirm source.metal result_directory");
    constexpr uint64_t words=8388608;
    constexpr unsigned seeds=7, rounds=3, warmups=16, repeats=5;
    NSString* folder=[NSString stringWithUTF8String:argv[2]];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    require(device!=nil && device.hasUnifiedMemory,"Apple Silicon required");
    NSError* error=nil;
    NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
        encoding:NSUTF8StringEncoding error:&error];
    require(source!=nil,"Read Metal source: "+errorText(error));
    MTLCompileOptions* options=[[MTLCompileOptions alloc]init];
    options.languageVersion=MTLLanguageVersion2_4;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    require(library!=nil,"Compile unchanged kernels: "+errorText(error));
    Pipelines pipelines(device,library);
    id<MTLCommandQueue> queue=[device newCommandQueue];require(queue!=nil,"No command queue");
    std::cout<<"GPU: "<<device.name.UTF8String<<". Verification des kernels inchanges..."<<std::endl;
    conformance(device,queue,pipelines,folder);
    std::cout<<"Precompilation de tous les pipelines, avant les mesures."<<std::endl;
    for(unsigned trial=0;trial<seeds;++trial)for(auto j:confirmJobs) {
        auto p=parameters(splitmix64(2000+trial),0,words,2048,j.workers,j.shift);
        auto pipeline=pipelines.get(j.engine,p);
        require(j.group<=pipeline.maxTotalThreadsPerThreadgroup,"Frozen group unsupported");
    }
    const NSUInteger preparedCount=pipelines.preparation.count;
    writeJSON(folder,@"pipeline-preparation-before-timing.json",pipelines.preparation);
    NSMutableArray<id<MTLBuffer>>* buffers=[NSMutableArray array];
    for(unsigned r=0;r<repeats;++r) {
        id<MTLBuffer> buffer=[device newBufferWithLength:(words+16)*8 options:MTLResourceStorageModeShared];
        require(buffer!=nil,"Cannot allocate five separate output buffers");[buffers addObject:buffer];
    }
    std::ofstream csv(std::string(argv[2])+"/measurements.csv");require(bool(csv),"Cannot open CSV");
    csv<<"ordinal,phase,trial,round,repeat,seed,engine,format,logical_lanes,draws_per_lane,physical_threads,threadgroup,words,gpu_seconds,wall_seconds,ns_per_word,stored_GB_per_second,thermal_before,thermal_after,full_comparison_pass\n";
    std::map<ConfirmFamily,std::vector<std::vector<double>>> samples;
    for(auto j:confirmJobs)samples[{j.engine,j.shift}].resize(seeds);
    require(samples.size()==22,"Duplicate frozen configuration");
    std::mt19937 order(20261023);
    uint64_t ordinal=0,compared=0;
    NSMutableArray* thermal=[NSMutableArray array];
    for(unsigned trial=0;trial<seeds;++trial) {
        const uint64_t seed=splitmix64(2000+trial);
        std::cout<<"Seed "<<trial+1<<'/'<<seeds<<": references CPU puis trois blocs de mesures."<<std::endl;
        std::map<unsigned,std::vector<uint64_t>> expected;
        for(unsigned e:{10u,1u,2u,7u})expected.emplace(e,laneReference(parameters(seed,0,words,2048,65536,0),e));
        for(unsigned round=0;round<rounds;++round) {
            std::vector<ConfirmJob> jobs(confirmJobs.begin(),confirmJobs.end());
            std::shuffle(jobs.begin(),jobs.end(),order);
            for(auto j:jobs) {
                const auto& oracle=expected.at(isMontmory(j.engine)?10:j.engine);
                const auto p=parameters(seed,0,words,2048,j.workers,j.shift);
                auto pipeline=pipelines.get(j.engine,p);
                require(pipelines.preparation.count==preparedCount,"Unexpected compilation during campaign");
                for(unsigned r=0;r<repeats;++r) {
                    auto* actual=static_cast<uint64_t*>(buffers[r].contents);
                    for(uint64_t i=0;i<words;++i)actual[i]=~(oracle[i]>>j.shift);
                    std::fill(actual+words,actual+words+16,UINT64_MAX);
                }
                const auto before=[NSProcessInfo processInfo].thermalState;
                std::vector<Times> warmTimes, measuredTimes;
                for(unsigned r=0;r<warmups;++r)warmTimes.push_back(dispatch(queue,pipeline,buffers[0],p,j.engine,j.group));
                // Five different buffers retain every timed result until CPU validation.
                for(unsigned r=0;r<repeats;++r)measuredTimes.push_back(dispatch(queue,pipeline,buffers[r],p,j.engine,j.group));
                const auto after=[NSProcessInfo processInfo].thermalState;
                for(unsigned r=0;r<repeats;++r) {compareBuffer(buffers[r],oracle,j);compared+=words;}
                auto writeRows=[&](const std::vector<Times>& times,bool warmup) {
                    for(unsigned r=0;r<times.size();++r) {
                        const auto t=times[r];const double ns=t.gpu*1e9/words;
                        require(t.group==j.group && std::isfinite(ns) && ns>0,"Invalid measurement");
                        csv<<ordinal++<<','<<(warmup?"warmup":"timed")<<','<<trial<<','<<round<<','<<r<<','<<seed<<','
                            <<engineNames[j.engine]<<','<<(j.shift?"mantissa53":"raw64")<<",2048,4096,"<<j.workers<<','<<j.group<<','<<words<<','
                            <<std::setprecision(15)<<t.gpu<<','<<t.wall<<','<<ns<<','<<8/ns<<','<<int(before)<<','<<int(after)<<','
                            <<(warmup?"not_individually_checked":"true")<<'\n';
                        if(!warmup)samples[{j.engine,j.shift}][trial].push_back(ns);
                    }
                };
                writeRows(warmTimes,true);writeRows(measuredTimes,false);
                [thermal addObject:@{@"trial":@(trial),@"round":@(round),@"job":configJSON(j),@"before":@(before),@"after":@(after)}];
            }
            csv.flush();require(bool(csv),"CSV write error");
            std::cout<<"  Bloc "<<round+1<<'/'<<rounds<<" termine, toutes les sorties chronometrees identiques."<<std::endl;
        }
    }
    NSMutableArray* summary=[NSMutableArray array];
    std::cout<<"\nConfigurations figees, sept nouvelles seeds, quinze mesures par seed.\n"
        <<"moteur                         format        GB/s   ratio/v3  seeds plus rapides\n";
    for(auto j:confirmJobs) {
        auto medians=confirmSeedMedians(samples.at({j.engine,j.shift}),seeds,rounds*repeats);
        auto base=confirmSeedMedians(samples.at({10,j.shift}),seeds,rounds*repeats);
        std::vector<double> ratios;unsigned faster=0;
        NSMutableArray* perSeed=[NSMutableArray array];
        for(unsigned i=0;i<seeds;++i) {
            ratios.push_back(base[i]/medians[i]);faster+=medians[i]<base[i];
            [perSeed addObject:@{@"seed":@(splitmix64(2000+i)),@"median_ns_per_word":@(medians[i]),@"paired_ratio_vs_v3":@(ratios.back())}];
        }
        const double ns=confirmMedian(medians),ratio=confirmMedian(ratios);
        std::cout<<std::left<<std::setw(31)<<engineNames[j.engine]<<std::setw(11)<<(j.shift?"mantissa53":"raw64")
            <<std::right<<std::fixed<<std::setprecision(3)<<std::setw(9)<<8/ns<<std::setw(11)<<ratio<<"  "<<faster<<'/'<<seeds<<'\n';
        NSMutableDictionary* item=[configJSON(j) mutableCopy];
        item[@"median_ns_per_word"]=@(ns);item[@"stored_GB_per_second"]=@(8/ns);
        item[@"median_paired_speedup_vs_v3"]=@(ratio);item[@"seeds_faster_than_v3"]=@(faster);item[@"per_seed"]=perSeed;
        [summary addObject:item];
    }
    require(compared==uint64_t(seeds)*rounds*confirmJobs.size()*repeats*words,"Comparison count mismatch");
    writeJSON(folder,@"summary.json",@{@"status":@"pass",@"device":device.name,@"results":summary,
        @"interpretation":@"Median of fifteen timings per seed, then median of seven seeds; paired ratios within seed. No row excluded. Warmups retained separately. No optimum claim."});
    writeJSON(folder,@"metadata.json",@{@"device":device.name,@"os":[NSProcessInfo processInfo].operatingSystemVersionString,
        @"fully_compared_timed_words":@(compared),@"pipeline_count_before_timing":@(preparedCount),@"pipeline_count_after_timing":@(pipelines.preparation.count),
        @"seeds":@(seeds),@"rounds":@(rounds),@"timed_repeats_per_job":@(repeats),@"warmup_dispatches_per_job":@(warmups),
        @"output_buffers":@(repeats),@"words_per_buffer":@(words),@"thermal":thermal});
    std::cout<<"PASS. Kernels inchanges, sorties chronometrees verifiees. Aucun envoi automatique."<<std::endl;
    return 0;
  }catch(const std::exception& e){std::cerr<<"ECHEC: "<<e.what()<<'\n';return 1;}
 }
}
