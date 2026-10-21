#include "v3_helpers.hpp"
#include "lane_reference.hpp"

struct LaneJob {unsigned engine,lanes,workers,group,shift;bool geometry;};
static double quantile(std::vector<double> values,double q) {
    std::sort(values.begin(),values.end());
    double pos=q*double(values.size()-1);size_t a=size_t(pos),b=std::min(a+1,values.size()-1);
    return values[a]+(pos-double(a))*(values[b]-values[a]);
}
static uint64_t compareAll(id<MTLBuffer> output,const std::vector<uint64_t>& expected,unsigned shift) {
    const uint64_t* actual=static_cast<const uint64_t*>(output.contents);
    for(size_t i=0;i<expected.size();++i)
        if(actual[i]!=(expected[i]>>shift)) throw std::runtime_error("Word mismatch at "+std::to_string(i));
    for(size_t i=expected.size();i<expected.size()+16;++i)
        require(actual[i]==UINT64_MAX,"Output guard overwritten");
    return expected.size();
}
int main(int argc,char**argv) {
 @autoreleasepool {
  try {
    require(argc==4,"usage: lanes source.metal results_directory repetitions");
    unsigned repetitions=unsigned(std::stoul(argv[3]));
    require(repetitions>=3 && repetitions<=31,"repetitions must be in [3,31]");
    const uint64_t words=8388608;
    NSString* folder=[NSString stringWithUTF8String:argv[2]];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    require(device!=nil && device.hasUnifiedMemory,"Apple Silicon Metal device required");
    NSError* error=nil;
    NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
    require(source!=nil,"Read Metal source: "+errorText(error));
    MTLCompileOptions* options=[[MTLCompileOptions alloc] init];options.languageVersion=MTLLanguageVersion2_4;
    std::cout<<"GPU: "<<device.name.UTF8String<<". Compilation des kernels v3 inchanges..."<<std::endl;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    require(library!=nil,"Compile Metal: "+errorText(error));
    Pipelines pipelines(device,library);
    id<MTLCommandQueue> queue=[device newCommandQueue];require(queue!=nil,"No command queue");
    conformance(device,queue,pipelines,folder);
    id<MTLBuffer> output=[device newBufferWithLength:(words+16)*sizeof(uint64_t) options:MTLResourceStorageModeShared];
    require(output!=nil,"Cannot allocate output");
    std::vector<LaneJob> jobs;unsigned skipped=0;
    auto add=[&](LaneJob j) {
        auto p=parameters(42,0,words,j.lanes,j.workers,j.shift);
        auto pipe=pipelines.get(j.engine,p);
        if(j.group>pipe.maxTotalThreadsPerThreadgroup) {++skipped;return;}
        jobs.push_back(j);
    };
    for(unsigned engine:{10u,1u,2u}) for(unsigned workers:{1024u,2048u,8192u,32768u,65536u,131072u,262144u})
        for(unsigned group:{64u,128u,256u,512u}) for(unsigned shift:{0u,11u})
            add({engine,2048,workers,group,shift,false});
    for(unsigned lanes:{256u,512u,1024u,4096u,8192u,16384u})
        for(unsigned group:{64u,128u,256u,512u}) for(unsigned shift:{0u,11u})
            add({10,lanes,65536,group,shift,true});
    require(!jobs.empty(),"No supported configurations");
    std::ofstream csv(std::string(argv[2])+"/measurements.csv");
    std::ofstream changes(std::string(argv[2])+"/logical-changes.csv");
    require(bool(csv)&&bool(changes),"Cannot open CSV files");
    csv<<"phase,engine,format,trial,seed,logical_lanes,draws_per_lane,physical_threads,requested_threadgroup,threadgroup,words,gpu_seconds,wall_seconds,ns_per_word,stored_GB_per_second,full_comparison_pass,thermal_state\n";
    changes<<"trial,seed,logical_lanes,draws_per_lane,different_words_from_T2048_D4096\n";
    using RefKey=std::pair<unsigned,unsigned>;
    using Group=std::tuple<unsigned,unsigned,unsigned,unsigned,unsigned,bool>;
    std::map<Group,std::vector<double>> measurements;
    std::mt19937 order(20261021);uint64_t compared=0;
    NSMutableArray* thermal=[NSMutableArray array];
    for(unsigned trial=0;trial<repetitions;++trial) {
        uint64_t seed=trial?splitmix64(trial):42;
        std::cout<<"Preparation des references CPU, seed "<<trial+1<<'/'<<repetitions<<"..."<<std::endl;
        std::map<RefKey,std::vector<uint64_t>> expected;
        for(unsigned lanes:{256u,512u,1024u,2048u,4096u,8192u,16384u})
            expected.emplace(RefKey(10,lanes),laneReference(parameters(seed,0,words,lanes,65536,0),10));
        for(unsigned e:{1u,2u}) expected.emplace(RefKey(e,2048),laneReference(parameters(seed,0,words,2048,65536,0),e));
        const auto& fixed=expected.at(RefKey(10,2048));
        for(unsigned lanes:{256u,512u,1024u,2048u,4096u,8192u,16384u}) {
            const auto& alternate=expected.at(RefKey(10,lanes));uint64_t different=0;
            for(uint64_t i=0;i<words;++i) different+=fixed[i]!=alternate[i];
            changes<<trial<<','<<seed<<','<<lanes<<','<<(words/lanes)<<','<<different<<'\n';
        }
        std::shuffle(jobs.begin(),jobs.end(),order);
        auto initialThermal=[NSProcessInfo processInfo].thermalState;
        for(auto j:jobs) {
            auto p=parameters(seed,0,words,j.lanes,j.workers,j.shift);
            auto pipe=pipelines.get(j.engine,p);
            const auto& oracle=expected.at(RefKey(j.engine,j.lanes));
            uint64_t* out=static_cast<uint64_t*>(output.contents);
            std::fill(out+words,out+words+16,UINT64_MAX);
            // On first use, every destination differs from its expected value.
            // A kernel which leaves holes cannot pass using an earlier job's data.
            if(trial==0) for(uint64_t i=0;i<words;++i) out[i]=~(oracle[i]>>j.shift);
            // First invocation of every configuration is fully checked before any
            // of its measured results can be accepted. Every timed result is also
            // checked in full, outside GPU and wall timers.
            dispatch(queue,pipe,output,p,j.engine,j.group);
            if(trial==0) compared+=compareAll(output,oracle,j.shift);
            dispatch(queue,pipe,output,p,j.engine,j.group);
            Times times=dispatch(queue,pipe,output,p,j.engine,j.group);
            compared+=compareAll(output,oracle,j.shift);
            const double ns=times.gpu*1e9/words;
            measurements[Group(j.engine,j.lanes,j.workers,unsigned(times.group),j.shift,j.geometry)].push_back(ns);
            csv<<(j.geometry?"logical_geometry_changes_stream":"physical_threads_same_stream")<<','<<engineNames[j.engine]<<','<<(j.shift?"mantissa53":"raw64")<<','<<trial<<','<<seed<<','<<j.lanes<<','<<(words/j.lanes)<<','<<j.workers<<','<<j.group<<','<<times.group<<','<<words<<','<<std::setprecision(12)<<times.gpu<<','<<times.wall<<','<<ns<<','<<(8.0/ns)<<",true,"<<int([NSProcessInfo processInfo].thermalState)<<'\n';
        }
        [thermal addObject:@{@"trial":@(trial),@"start":@(initialThermal),@"end":@([NSProcessInfo processInfo].thermalState)}];
        csv.flush();changes.flush();require(bool(csv)&&bool(changes),"CSV write failure");
        std::cout<<"Repetition "<<trial+1<<'/'<<repetitions<<" terminee; toutes les sorties comparees mot a mot."<<std::endl;
    }
    NSMutableArray* summary=[NSMutableArray array];
    using Best=std::tuple<unsigned,unsigned,unsigned,unsigned,bool>;
    std::map<Best,std::pair<Group,double>> best;
    for(const auto& entry:measurements) {
        auto [e,lanes,workers,group,shift,geometry]=entry.first;
        double ns=quantile(entry.second,.5);
        [summary addObject:@{@"engine":[NSString stringWithUTF8String:engineNames[e]],@"phase":geometry?@"logical_geometry_changes_stream":@"physical_threads_same_stream",@"format":shift?@"mantissa53":@"raw64",@"logical_lanes":@(lanes),@"draws_per_lane":@(words/lanes),@"physical_threads":@(workers),@"threadgroup":@(group),@"median_ns_per_word":@(ns),@"q10_ns_per_word":@(quantile(entry.second,.1)),@"q90_ns_per_word":@(quantile(entry.second,.9)),@"trials":@(entry.second.size())}];
        auto key=Best(e,lanes,workers,shift,geometry);
        if(!best.count(key)||ns<best.at(key).second) best[key]={entry.first,ns};
    }
    std::cout<<"\nA. Meme flux T=2048, D=4096 — mantissa53, meilleur groupe mesure par nombre de threads\n"
             <<"moteur                       threads groupe    ns/mot       GB/s\n";
    for(const auto& entry:best) {
        auto [e,lanes,workers,group,shift,geometry]=entry.second.first;(void)lanes;
        if(geometry||shift!=11)continue;
        double ns=entry.second.second;
        std::cout<<std::left<<std::setw(28)<<engineNames[e]<<std::right<<std::setw(8)<<workers<<std::setw(7)<<group<<std::setw(10)<<std::fixed<<std::setprecision(3)<<ns<<std::setw(11)<<8/ns<<'\n';
    }
    std::cout<<"\nB. Geometrie logique variable — Montmory, 65536 threads, volume fixe\n"
             <<"Ces changements de T et D changent le flux. Ils ne sont pas couverts par le TestU01 en cours.\n"
             <<"lanes   tirages/lane groupe    ns/mot       GB/s\n";
    for(const auto& entry:best) {
        auto [e,lanes,workers,group,shift,geometry]=entry.second.first;(void)geometry;
        if(e!=10||workers!=65536||shift!=11)continue;
        double ns=entry.second.second;
        std::cout<<std::setw(6)<<lanes<<std::setw(14)<<words/lanes<<std::setw(7)<<group<<std::setw(10)<<ns<<std::setw(11)<<8/ns<<'\n';
    }
    writeJSON(folder,@"summary.json",@{@"status":@"pass",@"device":device.name,@"results":summary});
    writeJSON(folder,@"pipeline-preparation.json",pipelines.preparation);
    writeJSON(folder,@"metadata.json",@{@"kit":@"montmory-lanes-v1",@"device":device.name,@"os":[NSProcessInfo processInfo].operatingSystemVersionString,@"words":@(words),@"repetitions":@(repetitions),@"configurations":@(jobs.size()),@"skipped_due_to_device_limit":@(skipped),@"fully_compared_words":@(compared),@"thermal_trials":thermal,@"timing":@"GPU timestamps and host dispatch/wait wall time; CPU reference generation and full comparisons excluded",@"fixed_contract":@"T=2048 D=4096 base=0; raw64 / word>>11",@"geometry_note":@"T and D vary jointly at fixed total words. This creates a different stream.",@"scope":@"Performance and bit conformance only. No TestU01 rerun. Best settings are measured choices, not universal optima."});
    std::cout<<"\nPASS. Details raw64 et mantissa53 : measurements.csv et summary.json.\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"ECHEC: "<<e.what()<<'\n';return 1;}
 }
}
