// mmc_gpu_wrap.mm — Host Metal + ring-buffer + TestU01 wrapper
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
extern "C" {
  double mmc_gpu_next01(void);
  void   mmc_gpu_init(uint64_t seed, uint32_t threads, uint32_t draws_per_thread);
}

static id<MTLDevice> gDev = nil;
static id<MTLCommandQueue> gQ = nil;
static id<MTLLibrary> gLib = nil;
static id<MTLComputePipelineState> gPSO = nil;
static id<MTLBuffer> gDevBuf = nil;
static id<MTLBuffer> gParamsBuf = nil;

typedef struct { uint64_t seed; uint64_t base; uint64_t draws; } Params;

/* ring buffer CPU */
static std::vector<double> gRing;
static size_t gHead=0, gTail=0, gCap=0;
static uint64_t gSeed=42;
static uint32_t gThreads=2048;
static uint32_t gDrawsPerThread=4096;
static uint64_t gBase=0;

static void refillGPU() {
  @autoreleasepool {
    // dispatch kernel to fill next chunk
    id<MTLCommandBuffer> cmd = [gQ commandBuffer];
    Params p = { gSeed, gBase, gDrawsPerThread };
    memcpy([gParamsBuf contents], &p, sizeof(Params));
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gPSO];
    [enc setBuffer:gDevBuf offset:0 atIndex:0];
    [enc setBuffer:gParamsBuf offset:0 atIndex:1];

    MTLSize tg = MTLSizeMake(gPSO.maxTotalThreadsPerThreadgroup,1,1);
    MTLSize ng = MTLSizeMake(gThreads,1,1);
    [enc dispatchThreads:ng threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // copy back to CPU ring
    size_t N = (size_t)gThreads*(size_t)gDrawsPerThread;
    double* src = (double*)[gDevBuf contents];
    for (size_t i=0;i<N;i++) {
      gRing[gHead] = src[i];
      gHead = (gHead+1) % gCap;
      if (gHead==gTail) gTail = (gTail+1) % gCap; // overwrite oldest if full
    }
    gBase += N;
  }
}

void mmc_gpu_init(uint64_t seed, uint32_t threads, uint32_t draws_per_thread){
  gSeed = seed; gThreads = threads; gDrawsPerThread = draws_per_thread;
  gDev = MTLCreateSystemDefaultDevice();
  gQ   = [gDev newCommandQueue];
  NSError* err = nil;

  // load library (assumes mmc_kernel.metallib embedded or default library)
  gLib = [gDev newDefaultLibrary];
  id<MTLFunction> fun = [gLib newFunctionWithName:@"mmc_fill"];
  gPSO = [gDev newComputePipelineStateWithFunction:fun error:&err];

  size_t N = (size_t)threads * (size_t)draws_per_thread;
  gDevBuf    = [gDev newBufferWithLength:N*sizeof(double) options:MTLResourceStorageModeShared];
  gParamsBuf = [gDev newBufferWithLength:sizeof(Params) options:MTLResourceStorageModeShared];

  // ring buffer 2× la taille d’un chunk pour amortir
  gCap = 2*N;
  gRing.assign(gCap, 0.0); gHead=0; gTail=0;

  // premier remplissage
  refillGPU();
}

double mmc_gpu_next01(void){
  if (gTail == gHead) {
    refillGPU(); // empty -> refill
  }
  double v = gRing[gTail];
  gTail = (gTail+1) % gCap;
  return v;
}
