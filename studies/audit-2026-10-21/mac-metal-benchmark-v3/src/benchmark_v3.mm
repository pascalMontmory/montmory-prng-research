#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include "reference.hpp"

static void require(bool ok,const std::string& message) {if(!ok) throw std::runtime_error(message);}
static std::string errorText(NSError* e) {return e?std::string(e.localizedDescription.UTF8String):"unknown Metal error";}
static void writeJSON(NSString* folder,NSString* name,id object) {
    NSError* error=nil;
    NSData* data=[NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingPrettyPrinted error:&error];
    require(data!=nil,"JSON: "+errorText(error));
    require([data writeToFile:[folder stringByAppendingPathComponent:name] options:NSDataWritingAtomic error:&error],"Write JSON: "+errorText(error));
}
struct Pipelines {
    id<MTLDevice> device; id<MTLLibrary> library;
    NSMutableDictionary* cache; NSMutableArray* preparation;
    Pipelines(id<MTLDevice> d,id<MTLLibrary> l):device(d),library(l),cache([NSMutableDictionary dictionary]),preparation([NSMutableArray array]) {}
    id<MTLComputePipelineState> get(unsigned engine,const BenchParams& p,bool reduce=false) {
        bool specialized=isSpecialized(engine);
        NSString* key=[NSString stringWithFormat:@"%u-%u-%u-%u-%u",engine,unsigned(reduce),specialized?p.r0:0,specialized?p.r1:0,specialized?p.r2:0];
        id<MTLComputePipelineState> previous=cache[key];if(previous) return previous;
        NSString* name=[NSString stringWithUTF8String:kernelNames[engine]];
        if(reduce) name=[name stringByAppendingString:@"_reduce"];
        NSError* error=nil;auto start=std::chrono::steady_clock::now();
        id<MTLFunction> function;
        if(specialized) {
            MTLFunctionConstantValues* values=[[MTLFunctionConstantValues alloc] init];
            [values setConstantValue:&p.r0 type:MTLDataTypeUInt atIndex:0];
            [values setConstantValue:&p.r1 type:MTLDataTypeUInt atIndex:1];
            [values setConstantValue:&p.r2 type:MTLDataTypeUInt atIndex:2];
            function=[library newFunctionWithName:name constantValues:values error:&error];
        } else function=[library newFunctionWithName:name];
        require(function!=nil,"Create kernel "+std::string(name.UTF8String)+": "+errorText(error));
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
        require(pipeline!=nil,"Compile pipeline: "+errorText(error));
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();cache[key]=pipeline;
        [preparation addObject:@{@"engine":[NSString stringWithUTF8String:engineNames[engine]],@"reduction":@(reduce),
            @"specialized_rotations":@(specialized),@"seed_first_use":@(p.seed),@"r0":@(p.r0),@"r1":@(p.r1),@"r2":@(p.r2),
            @"preparation_seconds":@(seconds),@"thread_execution_width":@(pipeline.threadExecutionWidth),
            @"max_threads_per_threadgroup":@(pipeline.maxTotalThreadsPerThreadgroup)}];return pipeline;
    }
};
struct Times {double gpu,wall;NSUInteger group;};
static Times dispatch(id<MTLCommandQueue> queue,id<MTLComputePipelineState> pipeline,id<MTLBuffer> output,
                      const BenchParams& p,unsigned engine,NSUInteger group=256) {
    auto start=std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command=[queue commandBuffer];require(command!=nil,"Cannot create command buffer");
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];require(encoder!=nil,"Cannot create encoder");
    [encoder setComputePipelineState:pipeline];[encoder setBuffer:output offset:0 atIndex:0];
    uint64_t legacy[]={p.seed,p.base,uint64_t(1)<<p.draws_shift};
    if(engine==3) [encoder setBytes:legacy length:sizeof(legacy) atIndex:1];else [encoder setBytes:&p length:sizeof(p) atIndex:1];
    NSUInteger workers=engine==3?p.logical_lanes:p.physical_threads;
    group=std::min({group,pipeline.maxTotalThreadsPerThreadgroup,workers});require(group>0,"Invalid group size");
    [encoder dispatchThreads:MTLSizeMake(workers,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"Metal command: "+errorText(command.error));
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    double gpu=command.GPUEndTime-command.GPUStartTime;require(gpu>0,"GPU timestamps unavailable");return {gpu,elapsed,group};
}
static uint64_t verifyOutput(id<MTLBuffer> output,const BenchParams& p,unsigned engine,bool full,bool reduce=false) {
    const uint64_t* words=static_cast<const uint64_t*>(output.contents);
    uint64_t count=reduce?p.physical_threads:p.words,checked=0;
    std::vector<uint64_t> reduced;if(reduce && full) reduced=referenceReduction(engine,p);
    auto one=[&](uint64_t i) {
        uint64_t expected=reduce?reduced[i]:referenceWord(engine,p,i);
        if(words[i]!=expected) {
            std::ostringstream msg;msg<<"Mismatch "<<engineNames[engine]<<" reduction="<<reduce<<" seed="<<p.seed<<" base="<<p.base<<" position="<<i;
            throw std::runtime_error(msg.str());
        }++checked;
    };
    if(full) for(uint64_t i=0;i<count;++i) one(i);
    else if(!reduce) {
        for(uint64_t i=0;i<16;++i) {one(i);one(count-1-i);}uint64_t index=42;
        for(unsigned j=0;j<128;++j) {index=splitmix64(index);one(index%count);}
    }
    for(uint64_t i=count;i<count+16;++i) require(words[i]==UINT64_MAX,"Output guard overwritten");
    return checked;
}
static void conformance(id<MTLDevice> device,id<MTLCommandQueue> queue,Pipelines& pipelines,NSString* folder) {
    auto kats=knownAnswers("reference/kat_vectors");std::array<uint32_t,36> inputs{};
    for(unsigned i=0;i<6;++i) std::copy(kats[i].input.begin(),kats[i].input.end(),inputs.begin()+6*i);
    id<MTLBuffer> in=[device newBufferWithBytes:inputs.data() length:sizeof(inputs) options:MTLResourceStorageModeShared];
    id<MTLBuffer> out=[device newBufferWithLength:24*sizeof(uint32_t) options:MTLResourceStorageModeShared];require(in && out,"Cannot allocate KAT buffers");
    NSError* error=nil;id<MTLFunction> f=[pipelines.library newFunctionWithName:@"reference_kat"];require(f!=nil,"Missing KAT kernel");
    id<MTLComputePipelineState> kat=[device newComputePipelineStateWithFunction:f error:&error];require(kat!=nil,"KAT pipeline: "+errorText(error));
    id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];require(command && encoder,"Cannot create KAT command");
    [encoder setComputePipelineState:kat];[encoder setBuffer:out offset:0 atIndex:0];[encoder setBuffer:in offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(6,1,1) threadsPerThreadgroup:MTLSizeMake(6,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"KAT command: "+errorText(command.error));
    const uint32_t* actual=static_cast<const uint32_t*>(out.contents);
    for(unsigned i=0;i<6;++i) for(unsigned j=0;j<4;++j) require(actual[4*i+j]==kats[i].expected[j],"Official GPU KAT mismatch");
    id<MTLBuffer> buffer=[device newBufferWithLength:(4096+16)*sizeof(uint64_t) options:MTLResourceStorageModeShared];require(buffer!=nil,"Cannot allocate conformance buffer");
    uint64_t checks=0,reductions=0;
    for(uint64_t seed:{UINT64_C(0),UINT64_C(1),UINT64_C(42),UINT64_MAX}) {
        std::cout<<"Verification seed "<<seed<<"..."<<std::endl;
        for(uint64_t n:{UINT64_C(32),UINT64_C(64),UINT64_C(2048)}) for(uint64_t base:{UINT64_C(0),n,UINT64_MAX-3}) {
            for(uint32_t workers:{7u,32u,256u,4096u}) for(NSUInteger group:{64ul,256ul})
            for(unsigned shift:{0u,11u}) for(unsigned engine=0;engine<engineCount;++engine) {
                if(engine==3 || (n<2048 && engine<8)) continue;
                BenchParams p=parameters(seed,base,n,32,workers,shift);
                for(bool reduce:{false,true}) {
                    if(reduce && !hasReduction(engine)) continue;
                    std::memset(buffer.contents,0xff,(4096+16)*sizeof(uint64_t));
                    dispatch(queue,pipelines.get(engine,p,reduce),buffer,p,engine,group);
                    uint64_t count=verifyOutput(buffer,p,engine,true,reduce);if(reduce) reductions+=count;else checks+=count;
                }
            }
            if(n==2048) {
                BenchParams p=parameters(seed,base,n,32,32,11);std::memset(buffer.contents,0xff,(4096+16)*sizeof(uint64_t));
                dispatch(queue,pipelines.get(3,p),buffer,p,3);checks+=verifyOutput(buffer,p,3,true);
            }
        }
    }
    writeJSON(folder,@"conformance.json",@{@"status":@"pass",@"bulk_words":@(checks),@"reduction_words":@(reductions),
        @"official_CPU_KATs":@6,@"official_GPU_KATs":@6,@"scope":@"Native Metal versus original C / official Random123; reductions checked per thread, including idle threads and guards."});
    std::cout<<"Conformite PASS: "<<checks<<" mots, "<<reductions<<" reductions; 6 vecteurs officiels CPU et GPU."<<std::endl;
}
struct Job {unsigned engine,shift,workers,group;bool reduce;};
static double median(std::vector<double> values) {std::sort(values.begin(),values.end());size_t n=values.size();return n%2?values[n/2]:(values[n/2-1]+values[n/2])/2;}
int main(int argc,char** argv) {
 @autoreleasepool {
  try {
    require(argc==5,"usage: benchmark source.metal result_directory words repetitions");
    uint64_t words=std::stoull(argv[3]);unsigned repetitions=unsigned(std::stoul(argv[4]));
    require(powerOfTwo(words) && words>=(1<<16) && words<=(1<<25) && repetitions>=3 && repetitions<=101,"words: power of two in [65536,33554432]; repetitions: [3,101]");
    NSString* folder=[NSString stringWithUTF8String:argv[2]];id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil && device.hasUnifiedMemory,"Apple Silicon Metal device required");
    NSError* error=nil;NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];require(source!=nil,"Read source: "+errorText(error));
    MTLCompileOptions* options=[[MTLCompileOptions alloc] init];options.languageVersion=MTLLanguageVersion2_4;
    std::cout<<"GPU: "<<device.name.UTF8String<<"\nCompilation Metal v3..."<<std::endl;
    auto compileStart=std::chrono::steady_clock::now();id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];require(library!=nil,"Compile Metal: "+errorText(error));
    double librarySeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-compileStart).count();Pipelines pipelines(device,library);
    id<MTLCommandQueue> queue=[device newCommandQueue];require(queue!=nil,"No command queue");auto thermalStart=[NSProcessInfo processInfo].thermalState;
    conformance(device,queue,pipelines,folder);
    uint64_t capacity=std::max(words,UINT64_C(262144))+16;require(capacity*sizeof(uint64_t)<=device.maxBufferLength,"Buffer exceeds device limit");
    id<MTLBuffer> output=[device newBufferWithLength:capacity*sizeof(uint64_t) options:MTLResourceStorageModeShared];require(output!=nil,"Cannot allocate output");
    std::vector<Job> jobs;
    for(unsigned workers:{8192u,65536u,262144u}) for(unsigned shift:{0u,11u}) for(unsigned e=0;e<engineCount;++e) if(e!=3) jobs.push_back({e,shift,workers,256,false});
    jobs.push_back({3,11,8192,256,false});
    for(unsigned group:{64u,128u}) for(unsigned shift:{0u,11u}) for(unsigned e:{1u,2u,6u,8u,9u,10u,11u,12u}) jobs.push_back({e,shift,65536,group,false});
    for(unsigned e=0;e<engineCount;++e) if(hasReduction(e)) jobs.push_back({e,0,65536,256,true});
    if(const char* requested=std::getenv("MONTMORY_CAPTURE")) {
        unsigned engine=engineCount;for(unsigned e=0;e<engineCount;++e) if(std::string(requested)==engineNames[e]) engine=e;require(engine<engineCount,"Unknown capture engine");
        BenchParams p=parameters(42,0,words,8192,engine==3?8192:65536,engine==3?11:0);auto pipeline=pipelines.get(engine,p);
        MTLCaptureManager* manager=[MTLCaptureManager sharedCaptureManager];require([manager supportsDestination:MTLCaptureDestinationGPUTraceDocument],"GPU capture requires Apple developer tool support");
        MTLCaptureDescriptor* descriptor=[[MTLCaptureDescriptor alloc] init];descriptor.captureObject=queue;descriptor.destination=MTLCaptureDestinationGPUTraceDocument;
        descriptor.outputURL=[NSURL fileURLWithPath:[folder stringByAppendingPathComponent:@"profile.gputrace"]];
        require([manager startCaptureWithDescriptor:descriptor error:&error],"GPU capture: "+errorText(error));dispatch(queue,pipeline,output,p,engine);[manager stopCapture];
        std::cout<<"Capture GPU enregistree hors chronometrage."<<std::endl;
    }
    std::ofstream csv(std::string(argv[2])+"/measurements.csv");require(bool(csv),"Cannot open measurements.csv");
    csv<<"engine,mode,format,physical_threads,requested_threadgroup,threadgroup,logical_lanes,draws_per_lane,trial,seed,generated_words,written_words,gpu_seconds,wall_seconds,ns_per_generated_word,generated_Gword_per_second,stored_GB_per_second,xor_checksum,sum_checksum\n";
    using Group=std::tuple<unsigned,unsigned,unsigned,unsigned,bool>;std::map<Group,std::vector<double>> measurements;
    using CheckKey=std::tuple<unsigned,unsigned,uint64_t>;std::map<CheckKey,uint64_t> xors,sums;
    NSMutableArray* thermal=[NSMutableArray array];std::mt19937 order(20261021);
    for(unsigned trial=0;trial<repetitions;++trial) {
        uint64_t seed=trial?splitmix64(trial):42;std::cout<<"Preparation et echauffement seed "<<trial+1<<'/'<<repetitions<<"..."<<std::endl;
        for(auto j:jobs) {
            BenchParams p=parameters(seed,0,words,8192,j.workers,j.shift);auto pipeline=pipelines.get(j.engine,p,j.reduce);
            for(unsigned warm=0;warm<2;++warm) dispatch(queue,pipeline,output,p,j.engine,j.group);
        }
        writeJSON(folder,@"pipeline-preparation.json",pipelines.preparation);std::shuffle(jobs.begin(),jobs.end(),order);auto trialThermal=[NSProcessInfo processInfo].thermalState;
        for(auto j:jobs) {
            BenchParams p=parameters(seed,0,words,8192,j.workers,j.shift);uint64_t count=j.reduce?p.physical_threads:words;
            uint64_t* data=static_cast<uint64_t*>(output.contents);std::fill(data+count,data+count+16,UINT64_MAX);
            Times t=dispatch(queue,pipelines.get(j.engine,p,j.reduce),output,p,j.engine,j.group);verifyOutput(output,p,j.engine,false,j.reduce);
            uint64_t sum=0,xorSum=0;for(uint64_t i=0;i<count;++i) {sum+=data[i];xorSum^=data[i];}
            auto key=CheckKey(isMontmory(j.engine)?0:j.engine,j.shift,seed);auto old=xors.find(key);
            if(old!=xors.end()) require(old->second==xorSum,"Bulk / reduction / partition XOR mismatch");else xors[key]=xorSum;
            if(!j.reduce) {auto prev=sums.find(key);if(prev!=sums.end()) require(prev->second==sum,"Bulk sum mismatch");else sums[key]=sum;}
            double ns=t.gpu*1e9/words;measurements[Group(j.engine,j.shift,j.workers,unsigned(t.group),j.reduce)].push_back(ns);
            csv<<engineNames[j.engine]<<','<<(j.reduce?"reduction_diagnostic":"bulk")<<','<<(j.shift?"mantissa53":"raw64")<<','<<j.workers<<','<<j.group<<','<<t.group<<",8192,"<<(words/8192)<<','<<trial<<','<<seed<<','<<words<<','<<count<<','
               <<std::setprecision(12)<<t.gpu<<','<<t.wall<<','<<ns<<','<<(1.0/ns)<<','<<(count*8.0/t.gpu/1e9)<<','<<std::hex<<xorSum<<','<<sum<<std::dec<<'\n';
        }
        [thermal addObject:@{@"trial":@(trial),@"start":@(trialThermal),@"end":@([NSProcessInfo processInfo].thermalState)}];csv.flush();require(bool(csv),"Cannot write measurements");std::cout<<"Repetition "<<trial+1<<'/'<<repetitions<<" terminee."<<std::endl;
    }
    NSMutableArray* summary=[NSMutableArray array];using Best=std::tuple<unsigned,unsigned,bool>;std::map<Best,std::pair<Group,double>> best;
    for(const auto& item:measurements) {
        auto [e,shift,workers,group,reduce]=item.first;const auto& v=item.second;double ns=median(v);
        [summary addObject:@{@"engine":[NSString stringWithUTF8String:engineNames[e]],@"mode":reduce?@"reduction_diagnostic":@"bulk",@"format":shift?@"mantissa53":@"raw64",@"physical_threads":@(workers),@"threadgroup":@(group),@"median_ns_per_generated_word":@(ns),@"min_ns_per_generated_word":@(*std::min_element(v.begin(),v.end())),@"max_ns_per_generated_word":@(*std::max_element(v.begin(),v.end())),@"trials":@(v.size())}];
        auto key=Best(e,shift,reduce);if(best.find(key)==best.end() || ns<best.at(key).second) best[key]={item.first,ns};
    }
    std::cout<<"\nGPU Metal v3 — meilleur reglage parmi les medianes mesurees; details dans measurements.csv\n"
             <<"moteur                      format       threads groupe   ns/mot       GB/s\n";
    for(const auto& item:best) {
        auto [e,shift,workers,group,reduce]=item.second.first;if(reduce) continue;double ns=item.second.second;
        std::cout<<std::left<<std::setw(28)<<engineNames[e]<<std::setw(13)<<(shift?"mantissa53":"raw64")<<std::right<<std::setw(7)<<workers<<std::setw(7)<<group<<std::setw(10)<<std::fixed<<std::setprecision(3)<<ns<<std::setw(11)<<(8.0/ns)<<'\n';
    }
    std::cout<<"\nDiagnostic reduction XOR : generation + accumulation, une ecriture par thread.\nCe mode ne livre pas le flux complet; unite Gmots generes/s, pas GB/s ecrits.\n";
    for(const auto& item:best) {
        auto [e,shift,workers,group,reduce]=item.second.first;(void)shift;(void)workers;(void)group;
        if(reduce) std::cout<<std::left<<std::setw(28)<<engineNames[e]<<std::right<<std::setw(10)<<(1.0/item.second.second)<<" Gmots/s\n";
    }
    writeJSON(folder,@"summary.json",@{@"status":@"pass",@"device":device.name,@"results":summary});
    writeJSON(folder,@"metadata.json",@{@"kit_version":@3,@"device":device.name,@"os":[NSProcessInfo processInfo].operatingSystemVersionString,
        @"words_per_run":@(words),@"repetitions":@(repetitions),@"configurations":@(jobs.size()),@"logical_lanes":@8192,@"draws_per_lane":@(words/8192),@"library_compile_seconds":@(librarySeconds),@"thermal_state_start":@(thermalStart),@"thermal_trials":thermal,
        @"profiling_run":@(std::getenv("MTL_CAPTURE_ENABLED")!=nullptr || std::getenv("MONTMORY_CAPTURE")!=nullptr),
        @"timing":@"GPU timestamps include computation and writes. CPU checks, pipeline specialization and compilation excluded. Wall time also recorded.",
        @"specialization":@"Only three seed-derived rotations for spec engines; keys remain runtime inputs. Costs in pipeline-preparation.json; caches keyed by rotation tuple and mode.",
        @"diagnostic":@"Reduction adds XOR accumulation and changes writes/register use. It is not pure compute. Do not subtract timings to infer a precise memory cost.",
        @"scope":@"Not a TestU01 battery. Best configurations are not universal optima. write_baseline_NOT_PRNG is not a random generator or bandwidth bound.",@"random123_commit":@"9545ff6413f258be2f04c1d319d99aaef7521150"});
    std::cout<<"\nPASS. Resultats: "<<argv[2]<<'\n';return 0;
  } catch(const std::exception& error) {std::cerr<<"ECHEC: "<<error.what()<<'\n';return 1;}
 }
}
