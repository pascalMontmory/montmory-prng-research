#include "v3_helpers.hpp"
#include "lane_reference.hpp"
#include "confirm_statistics.hpp"
#include <cmath>
struct Job {unsigned engine,shift,workers,group;};
using Family=std::pair<unsigned,unsigned>;
using Setting=std::tuple<unsigned,unsigned,unsigned,unsigned>;
static Setting setting(Job j){return {j.engine,j.shift,j.workers,j.group};}
static NSDictionary* jobJSON(Job j) {
    return @{@"engine":[NSString stringWithUTF8String:engineNames[j.engine]],
        @"format":j.shift?@"mantissa53":@"raw64",@"threads":@(j.workers),@"group":@(j.group)};
}
static void compare(id<MTLBuffer> buffer,const std::vector<uint64_t>& expected,Job j) {
    const auto* actual=static_cast<const uint64_t*>(buffer.contents);
    for(size_t i=0;i<expected.size();++i)
        if(actual[i]!=(expected[i]>>j.shift))throw std::runtime_error(std::string(engineNames[j.engine])+" mismatch at "+std::to_string(i));
    for(size_t i=expected.size();i<expected.size()+16;++i)require(actual[i]==UINT64_MAX,"Output guard overwritten");
}
int main(int argc,char** argv) {
 @autoreleasepool {
  try {
    require(argc==3,"usage: vector_benchmark all.metal results_directory");
    constexpr uint64_t words=8388608;
    constexpr unsigned searchSeeds=5,confirmSeeds=7,warmups=16;
    NSString* folder=[NSString stringWithUTF8String:argv[2]];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device && device.hasUnifiedMemory,"Apple Silicon required");
    NSError* error=nil;
    NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
    require(source!=nil,"Read Metal source: "+errorText(error));
    MTLCompileOptions* options=[[MTLCompileOptions alloc]init];options.languageVersion=MTLLanguageVersion2_4;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    require(library!=nil,"Metal compilation: "+errorText(error));
    Pipelines pipelines(device,library);id<MTLCommandQueue> queue=[device newCommandQueue];require(queue!=nil,"No command queue");
    std::cout<<"GPU: "<<device.name.UTF8String<<". Verification native des variantes..."<<std::endl;
    conformance(device,queue,pipelines,folder);
    std::vector<Job> candidates;
    for(unsigned e:{1u,2u,7u,10u,18u,20u,21u})for(unsigned shift:{0u,11u}) {
        auto p=parameters(42,0,words,2048,65536,shift);auto pipeline=pipelines.get(e,p);
        for(unsigned w:{8192u,65536u,131072u,262144u})for(unsigned g:{64u,128u,256u})
            if(g<=pipeline.maxTotalThreadsPerThreadgroup && g%pipeline.threadExecutionWidth==0)
                candidates.push_back({e,shift,w,g});
    }
    require(!candidates.empty(),"No supported candidate");
    const NSUInteger prepared=pipelines.preparation.count;
    writeJSON(folder,@"pipeline-preparation.json",pipelines.preparation);
    NSMutableArray<id<MTLBuffer>>* buffers=[NSMutableArray array];
    for(unsigned i=0;i<5;++i) {
        id<MTLBuffer> b=[device newBufferWithLength:(words+16)*8 options:MTLResourceStorageModeShared];
        require(b!=nil,"Cannot allocate output buffers");[buffers addObject:b];
    }
    std::ofstream csv(std::string(argv[2])+"/measurements.csv");require(bool(csv),"Cannot create CSV");
    csv<<"phase,trial,round,repeat,seed,engine,format,logical_lanes,draws_per_lane,physical_threads,threadgroup,words,gpu_seconds,wall_seconds,ns_per_word,thermal_before,thermal_after,full_comparison_pass\n";
    std::mt19937 order(20261024);uint64_t compared=0;
    auto measure=[&](Job j,uint64_t seed,const std::vector<uint64_t>& expected,const char* phase,unsigned trial,unsigned round,unsigned repeats) {
        auto p=parameters(seed,0,words,2048,j.workers,j.shift);
        auto pipeline=pipelines.get(j.engine,p);require(pipelines.preparation.count==prepared,"Compilation inside measurement campaign");
        for(unsigned r=0;r<repeats;++r) {
            auto* out=static_cast<uint64_t*>(buffers[r].contents);
            for(size_t i=0;i<words;++i)out[i]=~(expected[i]>>j.shift);
            std::fill(out+words,out+words+16,UINT64_MAX);
        }
        std::vector<Times> warm,timed;auto before=[NSProcessInfo processInfo].thermalState;
        for(unsigned r=0;r<warmups;++r)warm.push_back(dispatch(queue,pipeline,buffers[0],p,j.engine,j.group));
        for(unsigned r=0;r<repeats;++r)timed.push_back(dispatch(queue,pipeline,buffers[r],p,j.engine,j.group));
        auto after=[NSProcessInfo processInfo].thermalState;
        for(unsigned r=0;r<repeats;++r){compare(buffers[r],expected,j);compared+=words;}
        std::vector<double> result;
        for(bool isWarm:{true,false}) {
            const auto& times=isWarm?warm:timed;
            for(unsigned r=0;r<times.size();++r) {
                auto t=times[r];double ns=t.gpu*1e9/words;
                require(t.group==j.group && ns>0 && std::isfinite(ns),"Invalid timing");
                csv<<phase<<(isWarm?"_warmup":"_timed")<<','<<trial<<','<<round<<','<<r<<','<<seed<<','<<engineNames[j.engine]<<','
                   <<(j.shift?"mantissa53":"raw64")<<",2048,4096,"<<j.workers<<','<<j.group<<','<<words<<','<<std::setprecision(15)
                   <<t.gpu<<','<<t.wall<<','<<ns<<','<<int(before)<<','<<int(after)<<','<<(isWarm?"not_individually_checked":"true")<<'\n';
                if(!isWarm)result.push_back(ns);
            }
        }
        return result;
    };
    auto oracles=[&](uint64_t seed) {
        std::map<unsigned,std::vector<uint64_t>> result;
        for(unsigned e:{1u,2u,7u,10u})result.emplace(e,laneReference(parameters(seed,0,words,2048,65536,0),e));
        return result;
    };
    std::map<Setting,std::vector<double>> search;
    for(unsigned trial=0;trial<searchSeeds;++trial) {
        uint64_t seed=splitmix64(4000+trial);auto expected=oracles(seed);
        auto jobs=candidates;std::shuffle(jobs.begin(),jobs.end(),order);
        for(auto j:jobs) {
            auto v=measure(j,seed,expected.at(isMontmory(j.engine)?10:j.engine),"search",trial,0,3);
            search[setting(j)].push_back(confirmMedian(v));
        }
        csv.flush();require(bool(csv),"CSV failure");
        std::cout<<"Recherche "<<trial+1<<"/5 terminee, toutes les sorties identiques."<<std::endl;
    }
    std::map<Family,Job> chosen;
    for(auto j:candidates) {
        Family f={j.engine,j.shift};auto it=chosen.find(f);
        require(search.at(setting(j)).size()==searchSeeds,"Missing search data");
        if(it==chosen.end() || confirmMedian(search.at(setting(j)))<confirmMedian(search.at(setting(it->second))))chosen[f]=j;
    }
    require(chosen.size()==14,"Missing engine or format");
    NSMutableArray* selection=[NSMutableArray array];
    for(const auto& item:chosen)[selection addObject:jobJSON(item.second)];
    writeJSON(folder,@"selection-before-confirmation.json",selection);
    std::map<Family,std::vector<std::vector<double>>> samples;
    for(const auto& item:chosen)samples[item.first].resize(confirmSeeds);
    for(unsigned trial=0;trial<confirmSeeds;++trial) {
        uint64_t seed=splitmix64(5000+trial);auto expected=oracles(seed);
        for(unsigned round=0;round<3;++round) {
            std::vector<Job> jobs;for(const auto& item:chosen)jobs.push_back(item.second);
            std::shuffle(jobs.begin(),jobs.end(),order);
            for(auto j:jobs) {
                auto v=measure(j,seed,expected.at(isMontmory(j.engine)?10:j.engine),"confirm",trial,round,5);
                auto& target=samples.at({j.engine,j.shift})[trial];target.insert(target.end(),v.begin(),v.end());
            }
            csv.flush();require(bool(csv),"CSV failure");
        }
        std::cout<<"Confirmation "<<trial+1<<"/7 terminee, toutes les sorties identiques."<<std::endl;
    }
    NSMutableArray* summary=[NSMutableArray array];
    std::cout<<"\nConfirmation — memes T=2048,D=4096 ; 7 seeds nouvelles, 15 mesures par seed\n"
        <<"moteur                      format       threads groupe     GB/s   ratio/v3  seeds\n";
    for(const auto& item:chosen) {
        auto j=item.second;auto medians=confirmSeedMedians(samples.at(item.first),confirmSeeds,15);
        auto reference=confirmSeedMedians(samples.at({10,j.shift}),confirmSeeds,15);
        std::vector<double> ratios;unsigned wins=0;NSMutableArray* perSeed=[NSMutableArray array];
        for(unsigned i=0;i<confirmSeeds;++i) {
            ratios.push_back(reference[i]/medians[i]);wins+=medians[i]<reference[i];
            [perSeed addObject:@{@"seed":@(splitmix64(5000+i)),@"median_ns_per_word":@(medians[i]),@"paired_ratio_vs_v3":@(ratios.back())}];
        }
        double ns=confirmMedian(medians),ratio=confirmMedian(ratios);
        std::cout<<std::left<<std::setw(28)<<engineNames[j.engine]<<std::setw(11)<<(j.shift?"mantissa53":"raw64")<<std::right
            <<std::setw(9)<<j.workers<<std::setw(7)<<j.group<<std::fixed<<std::setprecision(3)<<std::setw(9)<<8/ns<<std::setw(11)<<ratio<<"  "<<wins<<"/7\n";
        auto result=[jobJSON(j) mutableCopy];result[@"median_ns_per_word"]=@(ns);result[@"stored_GB_per_second"]=@(8/ns);
        result[@"median_paired_ratio_vs_v3"]=@(ratio);result[@"seeds_faster_than_v3"]=@(wins);result[@"per_seed"]=perSeed;[summary addObject:result];
    }
    require(compared==(uint64_t(candidates.size())*searchSeeds*3+14*confirmSeeds*15)*words,"Word count mismatch");
    writeJSON(folder,@"summary.json",@{@"status":@"pass",@"device":device.name,@"results":summary,
        @"scope":@"Candidate benchmark; no automatic adoption, no optimum or statistical-quality ranking."});
    writeJSON(folder,@"metadata.json",@{@"device":device.name,@"os":[NSProcessInfo processInfo].operatingSystemVersionString,
        @"checked_timed_words":@(compared),@"search_configurations":@(candidates.size()),@"search_seeds":@5,@"confirmation_seeds":@7,
        @"logical_lanes":@2048,@"draws_per_lane":@4096,@"pipeline_count":@(prepared)});
    std::cout<<"PASS : flux conserve pour chaque sortie controlee. Aucun nouveau test statistique revendique."<<std::endl;
    return 0;
  }catch(const std::exception& e){std::cerr<<"ECHEC: "<<e.what()<<'\n';return 1;}
 }
}
