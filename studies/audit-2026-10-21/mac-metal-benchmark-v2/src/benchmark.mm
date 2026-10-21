#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include "reference.hpp"

static void require(bool ok,const std::string& message) {
    if(!ok) throw std::runtime_error(message);
}
static std::string errorText(NSError* error) {
    return error ? std::string(error.localizedDescription.UTF8String) : "unknown Metal error";
}
static void writeJSON(NSString* folder,NSString* name,id object) {
    NSError* error=nil;
    NSData* data=[NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingPrettyPrinted error:&error];
    require(data!=nil,"JSON: "+errorText(error));
    require([data writeToFile:[folder stringByAppendingPathComponent:name] options:NSDataWritingAtomic error:&error],
            "Write JSON: "+errorText(error));
}
struct Times { double gpu,wall; NSUInteger group; };

static Times dispatch(id<MTLCommandQueue> queue,id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> output,const BenchParams& p,unsigned engine,NSUInteger group=256) {
    auto start=std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command=[queue commandBuffer];
    require(command!=nil,"Cannot create command buffer");
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    require(encoder!=nil,"Cannot create compute encoder");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    uint64_t legacy[]={p.seed,p.base,uint64_t(1)<<p.draws_shift};
    if(engine==3) [encoder setBytes:legacy length:sizeof(legacy) atIndex:1];
    else [encoder setBytes:&p length:sizeof(p) atIndex:1];
    NSUInteger workers=engine==3?p.logical_lanes:p.physical_threads;
    group=std::min({group,pipeline.maxTotalThreadsPerThreadgroup,workers});
    require(group>0,"Invalid Metal threadgroup size");
    [encoder dispatchThreads:MTLSizeMake(workers,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"Metal command: "+errorText(command.error));
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    double gpu=command.GPUEndTime-command.GPUStartTime;
    require(gpu>0,"GPU timestamps unavailable: no timing result can be claimed");
    return {gpu,elapsed,group};
}

static uint64_t verifyOutput(id<MTLBuffer> output,const BenchParams& p,unsigned engine,bool full) {
    const uint64_t* words=static_cast<const uint64_t*>(output.contents);
    require(words!=nullptr,"Shared output buffer not accessible");
    uint64_t checked=0;
    auto one=[&](uint64_t i) {
        uint64_t expected=referenceWord(engine,p,i);
        if(words[i]!=expected) {
            std::ostringstream msg;
            msg<<"Mismatch "<<engineNames[engine]<<" seed="<<p.seed<<" base="<<p.base<<" position="<<i;
            throw std::runtime_error(msg.str());
        }
        ++checked;
    };
    if(full) for(uint64_t i=0;i<p.words;++i) one(i);
    else {
        for(uint64_t i=0;i<16;++i) { one(i); one(p.words-1-i); }
        uint64_t index=42;
        for(unsigned j=0;j<128;++j) { index=splitmix64(index); one(index%p.words); }
    }
    for(uint64_t i=p.words;i<p.words+16;++i) require(words[i]==UINT64_MAX,"Guard overwritten after output");
    return checked;
}

static uint64_t conformance(id<MTLDevice> device,id<MTLCommandQueue> queue,NSArray* pipelines) {
    auto kats=knownAnswers("reference/kat_vectors");
    std::array<uint32_t,36> inputs{};
    for(unsigned i=0;i<6;++i) std::copy(kats[i].input.begin(),kats[i].input.end(),inputs.begin()+6*i);
    id<MTLBuffer> in=[device newBufferWithBytes:inputs.data() length:sizeof(inputs) options:MTLResourceStorageModeShared];
    id<MTLBuffer> out=[device newBufferWithLength:24*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    require(in && out,"Cannot allocate known-answer buffers");
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    require(command && encoder,"Cannot create known-answer command");
    [encoder setComputePipelineState:pipelines[8]];
    [encoder setBuffer:out offset:0 atIndex:0]; [encoder setBuffer:in offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(6,1,1) threadsPerThreadgroup:MTLSizeMake(6,1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"Known-answer GPU command: "+errorText(command.error));
    const uint32_t* actual=static_cast<const uint32_t*>(out.contents);
    for(unsigned i=0;i<6;++i) for(unsigned j=0;j<4;++j)
        require(actual[4*i+j]==kats[i].expected[j],"GPU official known-answer mismatch");
    const uint64_t n=2048;
    id<MTLBuffer> buffer=[device newBufferWithLength:(n+16)*sizeof(uint64_t) options:MTLResourceStorageModeShared];
    require(buffer!=nil,"Cannot allocate conformance buffer");
    uint64_t checks=0;
    for(uint64_t seed:{UINT64_C(0),UINT64_C(1),UINT64_C(42),UINT64_MAX})
    for(uint64_t base:{UINT64_C(0),n,UINT64_MAX-3}) {
        for(uint32_t workers:{7u,32u,64u,256u,257u,4096u}) for(NSUInteger group:{64ul,256ul})
        for(unsigned shift:{0u,11u}) for(unsigned engine=0;engine<8;++engine) {
            if(engine==3) continue; // Original kernel only emits mantissas.
            BenchParams p=parameters(seed,base,n,32,workers,shift);
            std::memset(buffer.contents,0xff,(n+16)*sizeof(uint64_t));
            dispatch(queue,pipelines[engine],buffer,p,engine,group);
            checks+=verifyOutput(buffer,p,engine,true);
        }
        BenchParams p=parameters(seed,base,n,32,32,11);
        std::memset(buffer.contents,0xff,(n+16)*sizeof(uint64_t));
        dispatch(queue,pipelines[3],buffer,p,3);
        checks+=verifyOutput(buffer,p,3,true);
    }
    return checks;
}

struct Job { unsigned engine,shift,workers; };
static double median(std::vector<double> values) {
    std::sort(values.begin(),values.end()); size_t n=values.size();
    return n%2?values[n/2]:(values[n/2-1]+values[n/2])/2;
}

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            require(argc==5,"usage: benchmark source.metal result_directory words repetitions");
            uint64_t words=std::stoull(argv[3]); unsigned repetitions=unsigned(std::stoul(argv[4]));
            require(powerOfTwo(words) && words>=(1<<16) && words<=(1<<25) && repetitions>=3 && repetitions<=101,
                    "words must be a power of two in [65536,33554432]; repetitions in [3,101]");
            NSString* folder=[NSString stringWithUTF8String:argv[2]];
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            require(device!=nil,"No Metal device");
            require(device.hasUnifiedMemory,"This kit targets Apple Silicon unified memory");
            NSError* error=nil;
            NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                encoding:NSUTF8StringEncoding error:&error];
            require(source!=nil,"Read Metal source: "+errorText(error));
            MTLCompileOptions* options=[[MTLCompileOptions alloc] init];
            options.languageVersion=MTLLanguageVersion2_4;
            std::cout<<"GPU: "<<device.name.UTF8String<<"\nCompilation Metal..."<<std::endl;
            id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
            require(library!=nil,"Compile Metal: "+errorText(error));
            NSMutableArray* pipelines=[NSMutableArray array];
            for(NSString* name in @[@"montmory_indexed",@"threefry_bulk",@"philox_bulk",@"mmc_fill",
                                   @"montmory_u32_indexed",@"montmory_u64_tiled",@"montmory_u32_tiled",
                                   @"write_baseline",@"reference_kat"]) {
                id<MTLFunction> f=[library newFunctionWithName:name];
                require(f!=nil,"Missing kernel");
                id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&error];
                require(p!=nil,"Compile pipeline: "+errorText(error));
                [pipelines addObject:p];
            }
            id<MTLCommandQueue> queue=[device newCommandQueue]; require(queue!=nil,"No command queue");
            auto thermalStart=[NSProcessInfo processInfo].thermalState;
            uint64_t exact=conformance(device,queue,pipelines);
            writeJSON(folder,@"conformance.json",@{@"status":@"pass",@"device":device.name,
                @"exact_words":@(exact),@"official_CPU_KATs":@6,@"official_GPU_KATs":@6,
                @"scope":@"Native Metal versus original C and official Random123, including guard regions"});
            std::cout<<"Conformite PASS: "<<exact<<" mots; 6 vecteurs officiels CPU et GPU.\n";
            require((words+16)*sizeof(uint64_t)<=device.maxBufferLength,"Requested buffer exceeds device limit");
            id<MTLBuffer> output=[device newBufferWithLength:(words+16)*sizeof(uint64_t) options:MTLResourceStorageModeShared];
            require(output!=nil,"Cannot allocate benchmark output");
            std::vector<Job> jobs;
            for(unsigned workers:{8192u,65536u,262144u}) for(unsigned shift:{0u,11u}) for(unsigned engine=0;engine<8;++engine)
                if(engine!=3) jobs.push_back({engine,shift,workers});
            jobs.push_back({3,11,8192});
            for(auto j:jobs) for(unsigned warm=0;warm<2;++warm) {
                BenchParams p=parameters(42,0,words,8192,j.workers,j.shift);
                dispatch(queue,pipelines[j.engine],output,p,j.engine);
            }
            std::ofstream csv(std::string(argv[2])+"/measurements.csv");
            require(bool(csv),"Cannot open measurements.csv");
            csv<<"engine,format,physical_threads,threadgroup,logical_lanes,draws_per_lane,trial,seed,words,gpu_seconds,wall_seconds,ns_per_word,GB_per_second,checksum\n";
            using Group=std::tuple<unsigned,unsigned,unsigned>;
            std::map<Group,std::vector<double>> measurements;
            using CheckKey=std::tuple<unsigned,unsigned,uint64_t>;
            std::map<CheckKey,uint64_t> sums;
            std::mt19937 order(20261021);
            for(unsigned trial=0;trial<repetitions;++trial) {
                uint64_t seed=trial?splitmix64(trial):42;
                std::shuffle(jobs.begin(),jobs.end(),order);
                for(auto j:jobs) {
                    BenchParams p=parameters(seed,0,words,8192,j.workers,j.shift);
                    uint64_t* data=static_cast<uint64_t*>(output.contents);
                    std::fill(data+words,data+words+16,UINT64_MAX);
                    Times t=dispatch(queue,pipelines[j.engine],output,p,j.engine);
                    verifyOutput(output,p,j.engine,false);
                    uint64_t sum=0; for(uint64_t i=0;i<words;++i) sum+=data[i];
                    auto key=CheckKey(isMontmory(j.engine)?0:j.engine,j.shift,seed);
                    auto old=sums.find(key);
                    if(old!=sums.end()) require(old->second==sum,"Partition or historical-kernel checksum mismatch");
                    else sums[key]=sum;
                    double ns=t.gpu*1e9/words;
                    measurements[Group(j.engine,j.shift,j.workers)].push_back(ns);
                    csv<<engineNames[j.engine]<<','<<(j.shift?"mantissa53":"raw64")<<','<<j.workers<<','<<t.group
                       <<",8192,"<<(words/8192)<<','<<trial<<','<<seed<<','<<words<<','
                       <<std::setprecision(12)<<t.gpu<<','<<t.wall<<','<<ns<<','<<(words*8.0/t.gpu/1e9)<<','
                       <<std::hex<<sum<<std::dec<<'\n';
                }
                csv.flush(); require(bool(csv),"Cannot write measurements");
                std::cout<<"Repetition "<<trial+1<<'/'<<repetitions<<" terminee."<<std::endl;
            }
            NSMutableArray* summary=[NSMutableArray array];
            std::cout<<"\nGPU Metal — medianes; un mot = 64 bits stockes; plus petit = plus rapide\n"
                     <<"moteur                 format       threads    ns/mot       GB/s\n";
            for(const auto& item:measurements) {
                auto [engine,shift,workers]=item.first; const auto& times=item.second;
                double ns=median(times);
                std::cout<<std::left<<std::setw(23)<<engineNames[engine]<<std::setw(13)<<(shift?"mantissa53":"raw64")
                         <<std::right<<std::setw(7)<<workers<<std::setw(11)<<std::fixed<<std::setprecision(3)<<ns
                         <<std::setw(11)<<(8.0/ns)<<'\n';
                [summary addObject:@{@"engine":[NSString stringWithUTF8String:engineNames[engine]],
                    @"format":shift?@"mantissa53":@"raw64",@"physical_threads":@(workers),
                    @"median_ns_per_word":@(ns),@"min_ns_per_word":@(*std::min_element(times.begin(),times.end())),
                    @"max_ns_per_word":@(*std::max_element(times.begin(),times.end())),
                    @"median_GB_per_second":@(8.0/ns),@"trials":@(times.size())}];
            }
            writeJSON(folder,@"summary.json",@{@"status":@"pass",@"device":device.name,@"results":summary});
            NSMutableArray* pipelineInfo=[NSMutableArray array];
            for(unsigned engine=0;engine<8;++engine) {
                id<MTLComputePipelineState> pipeline=pipelines[engine];
                [pipelineInfo addObject:@{@"engine":[NSString stringWithUTF8String:engineNames[engine]],
                    @"thread_execution_width":@(pipeline.threadExecutionWidth),
                    @"max_threads_per_threadgroup":@(pipeline.maxTotalThreadsPerThreadgroup)}];
            }
            writeJSON(folder,@"metadata.json",@{@"device":device.name,@"kit_version":@2,
                @"pipelines":pipelineInfo,
                @"os":[NSProcessInfo processInfo].operatingSystemVersionString,
                @"words_per_run":@(words),@"repetitions":@(repetitions),@"logical_lanes":@8192,
                @"draws_per_lane":@(words/8192),@"random123_commit":@"9545ff6413f258be2f04c1d319d99aaef7521150",
                @"thermal_state_start":@(thermalStart),@"thermal_state_end":@([NSProcessInfo processInfo].thermalState),
                @"timing":@"GPU command-buffer timestamps; output writes included; CPU preparation, compile, checksum excluded; wall time also recorded",
                @"comparison":@"All Philox output words consumed. Indexed engines precompute keys outside timing. Historical kernel includes initialization.",
                @"scope":@"Native Metal conformance and throughput; not a TestU01 battery. write_baseline_NOT_PRNG is only a simple write reference, not a random generator or a hardware bandwidth bound."});
            std::cout<<"\nPASS. Resultats: "<<argv[2]<<"\n";
            return 0;
        } catch(const std::exception& error) {
            std::cerr<<"ECHEC: "<<error.what()<<"\n"; return 1;
        }
    }
}
