// Reuse the same pipeline factory, conformance checks and dispatch as the benchmark.
#define main unused_benchmark_main
#include "benchmark_v3.mm"
#undef main
extern "C" {
#include "unif01.h"
#include "bbattery.h"
}
static id<MTLCommandQueue> testQueue;
static id<MTLComputePipelineState> testPipeline;
static id<MTLBuffer> testBuffer;
static BenchParams testParams;
static unsigned testEngine;
static uint64_t testPosition;
static double nextUniform() {
    if(testPosition==testParams.words) {
        @autoreleasepool {
            uint64_t* data=static_cast<uint64_t*>(testBuffer.contents);
            std::fill(data+testParams.words,data+testParams.words+16,UINT64_MAX);
            dispatch(testQueue,testPipeline,testBuffer,testParams,testEngine);
            verifyOutput(testBuffer,testParams,testEngine,testParams.base==0);
            testParams.base+=testParams.words;testPosition=0;
        }
    }
    const uint64_t* data=static_cast<const uint64_t*>(testBuffer.contents);
    return double(data[testPosition++])*0x1p-53;
}
int main(int argc,char** argv) {
 @autoreleasepool {
  try {
    require(argc==4,"usage: testu01_metal engine small|crush|big results_directory");
    std::string engineName=argv[1],battery=argv[2];testEngine=engineCount;
    for(unsigned e=0;e<engineCount;++e) if(engineName==engineNames[e] && isMontmory(e)) testEngine=e;
    require(testEngine<engineCount,"Choose a Montmory engine from the benchmark");
    require(battery=="small" || battery=="crush" || battery=="big","Unknown battery");
    NSString* folder=[NSString stringWithUTF8String:argv[3]];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil && device.hasUnifiedMemory,"Apple Silicon device required");
    NSError* error=nil;NSString* source=[NSString stringWithContentsOfFile:@"generated/all.metal" encoding:NSUTF8StringEncoding error:&error];require(source!=nil,"Read Metal source");
    MTLCompileOptions* options=[[MTLCompileOptions alloc] init];options.languageVersion=MTLLanguageVersion2_4;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];require(library!=nil,"Metal compile: "+errorText(error));
    Pipelines pipelines(device,library);testQueue=[device newCommandQueue];require(testQueue!=nil,"No queue");
    conformance(device,testQueue,pipelines,folder);
    // Historical statistical stream: logical geometry differs from performance runs.
    testParams=parameters(42,0,UINT64_C(2048)*4096,2048,testEngine==3?2048:65536,11);
    testPosition=testParams.words;testPipeline=pipelines.get(testEngine,testParams);
    testBuffer=[device newBufferWithLength:(testParams.words+16)*sizeof(uint64_t) options:MTLResourceStorageModeShared];require(testBuffer!=nil,"Cannot allocate TestU01 output");
    writeJSON(folder,@"testu01-metadata.json",@{@"device":device.name,@"engine":[NSString stringWithUTF8String:argv[1]],
        @"battery":[NSString stringWithUTF8String:argv[2]],@"seed":@42,@"logical_lanes":@2048,@"draws_per_lane":@4096,
        @"physical_threads":@(testParams.physical_threads),@"output":@"53-bit uniforms in historical order",
        @"os":[NSProcessInfo processInfo].operatingSystemVersionString});
    std::string name="Montmory v3 native Metal "+engineName;
    unif01_Gen* generator=unif01_CreateExternGen01(name.data(),nextUniform);
    std::cout<<"Native Metal TestU01: "<<engineName<<" battery="<<battery<<" seed=42 lanes=2048 draws=4096\n";
    if(battery=="small") bbattery_SmallCrush(generator);
    else if(battery=="crush") bbattery_Crush(generator);
    else bbattery_BigCrush(generator);
    unif01_DeleteExternGen01(generator);return 0;
  } catch(const std::exception& error) {std::cerr<<"ECHEC: "<<error.what()<<'\n';return 1;}
 }
}
