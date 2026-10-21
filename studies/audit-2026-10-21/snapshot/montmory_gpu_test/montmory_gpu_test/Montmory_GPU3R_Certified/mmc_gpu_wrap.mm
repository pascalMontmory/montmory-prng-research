#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <stdint.h>
#include <string.h>
typedef struct { uint64_t seed; uint64_t base; uint64_t draws; } Params;extern "C" {
extern "C" double mmc_gpu_next01(void);
extern "C" void mmc_gpu_init(uint64_t seed, uint32_t threads, uint32_t draws_per_thread);
}
static id<MTLDevice> gDev; static id<MTLCommandQueue> gQ; static id<MTLLibrary> gLib; static id<MTLComputePipelineState> gPSO;
static id<MTLBuffer> gDevBuf; static id<MTLBuffer> gParamsBuf;
static std::vector<double> gRing; static size_t gHead=0,gTail=0,gCap=0;
static uint64_t gSeed=42,gBase=0; static uint32_t gThreads=2048,gDpt=4096;

static void refillGPU() {
  @autoreleasepool {
    id<MTLCommandBuffer> cmd = [gQ commandBuffer];
    Params p = { gSeed, gBase, gDpt };
    memcpy([gParamsBuf contents], &p, sizeof(Params));
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gPSO];
    [enc setBuffer:gDevBuf offset:0 atIndex:0];
    [enc setBuffer:gParamsBuf offset:0 atIndex:1];
    MTLSize tg = MTLSizeMake(gPSO.maxTotalThreadsPerThreadgroup,1,1);
    MTLSize ng = MTLSizeMake(gThreads,1,1);
    [enc dispatchThreads:ng threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];

    size_t N = (size_t)gThreads * (size_t)gDpt;
    uint64_t* src = (uint64_t*)[gDevBuf contents];
    for (size_t i=0;i<N;i++){ gRing[gHead] = (double)src[i] * (1.0/9007199254740992.0); gHead=(gHead+1)%gCap; if (gHead==gTail) gTail=(gTail+1)%gCap; }
    gBase += N;
  }
}
extern "C" void mmc_gpu_init(uint64_t seed, uint32_t threads, uint32_t draws_per_thread){
  gSeed=seed; gThreads=threads; gDpt=draws_per_thread;
  gDev = MTLCreateSystemDefaultDevice(); if(!gDev){ fprintf(stderr,"No Metal device\n"); exit(1); }
  gQ   = [gDev newCommandQueue];
  NSError *err = nil;
  gLib = [gDev newLibraryWithFile:@"build/mmc_kernel.metallib" error:&err];
  if(!gLib){ fprintf(stderr, "metallib load error\n"); exit(1); }
  id<MTLFunction> fun = [gLib newFunctionWithName:@"mmc_fill"];
  gPSO = [gDev newComputePipelineStateWithFunction:fun error:&err];
  if(!gPSO){ fprintf(stderr,"PSO error\n"); exit(1); }
  size_t N=(size_t)threads*(size_t)draws_per_thread;
  gDevBuf    = [gDev newBufferWithLength:N*sizeof(uint64_t) options:MTLResourceStorageModeShared];
  gParamsBuf = [gDev newBufferWithLength:sizeof(Params) options:MTLResourceStorageModeShared];
  gCap = 2*N; gRing.assign(gCap, 0.0); gHead=gTail=0; gBase=0;
  refillGPU();
}
extern "C" double mmc_gpu_next01(void){
  if (gTail==gHead) refillGPU();
  double v = gRing[gTail]; gTail=(gTail+1)%gCap; return v;
}
