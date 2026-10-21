// Native Mac check of the recovered Metal kernel against the recovered C core.
// Prepared on Linux; must be compiled and executed on a Mac with Metal.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include "mmc_core.c"

typedef struct { uint64_t seed, base, draws; } Params;
static_assert(sizeof(Params)==24, "kernel parameter layout");

int main(int argc, char **argv) {
    if(argc!=2) { fprintf(stderr,"usage: check_metal kernel.metallib\n"); return 2; }
    @autoreleasepool {
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if(!device) { fprintf(stderr,"No Metal device\n"); return 2; }
        NSError *error=nil;
        NSURL *url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library=[device newLibraryWithURL:url error:&error];
        if(!library) { fprintf(stderr,"Library: %s\n",error.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> kernel=[library newFunctionWithName:@"mmc_fill"];
        if(!kernel) { fprintf(stderr,"Missing kernel\n"); return 2; }
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:kernel error:&error];
        if(!pipeline) { fprintf(stderr,"Pipeline: %s\n",error.localizedDescription.UTF8String); return 2; }
        id<MTLCommandQueue> queue=[device newCommandQueue];
        if(!queue) return 2;
        const uint64_t seeds[]={0,1,42,UINT64_MAX};
        const uint32_t configs[][2]={{1,1},{2,3},{7,11},{32,16},{8192,2}};
        uint64_t checks=0;
        for(unsigned a=0;a<4;++a) for(unsigned c=0;c<5;++c) {
            uint32_t threads=configs[c][0], draws=configs[c][1];
            uint64_t count=(uint64_t)threads*draws;
            const uint64_t bases[]={0,count,UINT64_MAX-3};
            id<MTLBuffer> out=[device newBufferWithLength:count*sizeof(uint64_t) options:MTLResourceStorageModeShared];
            if(!out) return 2;
            for(unsigned b=0;b<3;++b) {
                Params p={seeds[a],bases[b],draws};
                id<MTLCommandBuffer> cmd=[queue commandBuffer];
                id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
                if(!cmd || !enc) return 2;
                [enc setComputePipelineState:pipeline];
                [enc setBuffer:out offset:0 atIndex:0];
                [enc setBytes:&p length:sizeof(p) atIndex:1];
                NSUInteger width=MIN((NSUInteger)threads,pipeline.maxTotalThreadsPerThreadgroup);
                [enc dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
                [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                if(cmd.status!=MTLCommandBufferStatusCompleted) {
                    fprintf(stderr,"Command: %s\n",cmd.error.localizedDescription.UTF8String); return 2;
                }
                const uint64_t *words=(const uint64_t *)out.contents;
                for(uint32_t lane=0;lane<threads;++lane) {
                    mmc_state_t st; mmc_init(&st,p.seed,lane);
                    st.s += (p.base+(uint64_t)lane*draws)*st.gamma;
                    for(uint32_t i=0;i<draws;++i) {
                        uint64_t expected=mmc_next_u64(&st)>>11;
                        if(words[(uint64_t)lane*draws+i]!=expected) {
                            fprintf(stderr,"Mismatch seed=%llu base=%llu lane=%u draw=%u\n",
                                    (unsigned long long)p.seed,(unsigned long long)p.base,lane,i);
                            return 1;
                        }
                        ++checks;
                    }
                }
            }
        }
        NSDictionary *report=@{@"status":@"pass",@"device":device.name,
                               @"mantissas_compared":@(checks),@"scope":@"Recovered Metal kernel versus native CPU, 53 bits"};
        NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
        if(!json) return 2;
        fwrite(json.bytes,1,json.length,stdout); puts("");
    }
    return 0;
}
