#include <metal_stdlib>
using namespace metal;
#include <Random123/threefry.h>
#include <Random123/philox.h>
#include "bench_params.h"
// Includes the original kernel unchanged, plus its integer helper functions.
#include "mmc_kernel.metal"
#include "montmory_u32.metalh"
#include "montmory_batch.metalh"

// Same logical stream as the original kernel; only assignment to GPU threads
// changes. Keys are prepared before timing, as for the two Random123 engines.
kernel void montmory_indexed(device ulong* out [[buffer(0)]],
                            constant BenchParams& p [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
    MmcState st;
    st.k0=p.k0; st.k1=p.k1; st.k2=p.k2;
    st.r0=p.r0; st.r1=p.r1; st.r2=p.r2;
    for (ulong i=tid; i<p.words; i+=p.physical_threads) {
        ulong lane=(i>>p.draws_shift)&(p.logical_lanes-1);
        ulong gamma=0x9E3779B97F4A7C15UL+2*lane;
        ulong counter=p.seed+(p.base+i+1)*gamma;
        out[i]=pcg_dxsm(perm3(counter,st))>>p.output_shift;
    }
}

kernel void threefry_bulk(device ulong* out [[buffer(0)]],
                         constant BenchParams& p [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
    threefry2x32_key_t key={{p.key0,p.key1}};
    for (ulong i=tid; i<p.words; i+=p.physical_threads) {
        ulong index=p.base+i;
        threefry2x32_ctr_t counter={{uint(index),uint(index>>32)}};
        threefry2x32_ctr_t value=threefry2x32_R(20,counter,key);
        out[i]=(ulong(value.v[0])|(ulong(value.v[1])<<32))>>p.output_shift;
    }
}

kernel void philox_bulk(device ulong* out [[buffer(0)]],
                       constant BenchParams& p [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
    philox4x32_key_t key={{p.key0,p.key1}};
    for (ulong i=tid; i<p.words/2; i+=p.physical_threads) {
        ulong index=p.base+i;
        philox4x32_ctr_t counter={{uint(index),uint(index>>32),0,0}};
        philox4x32_ctr_t value=philox4x32_R(10,counter,key);
        // Consume both 64-bit words returned by every Philox block.
        out[2*i]=(ulong(value.v[0])|(ulong(value.v[1])<<32))>>p.output_shift;
        out[2*i+1]=(ulong(value.v[2])|(ulong(value.v[3])<<32))>>p.output_shift;
    }
}

kernel void montmory_u32_indexed(device ulong* out [[buffer(0)]],
                                constant BenchParams& p [[buffer(1)]],
                                uint tid [[thread_position_in_grid]]) {
    mmc_indexed_body<true>(out,p,tid);
}

kernel void montmory_u64_tiled(device ulong* out [[buffer(0)]],
                              constant BenchParams& p [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {
    mmc_tiled_body<false>(out,p,tid);
}

kernel void montmory_u32_tiled(device ulong* out [[buffer(0)]],
                              constant BenchParams& p [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {
    mmc_tiled_body<true>(out,p,tid);
}

// Timing reference for simple writes. This is deliberately not a PRNG.
kernel void write_baseline(device ulong* out [[buffer(0)]],
                           constant BenchParams& p [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {
    for(ulong i=tid;i<p.words;i+=p.physical_threads)
        out[i]=((p.base+i)^p.seed)>>p.output_shift;
}

kernel void reference_kat(device uint* out [[buffer(0)]],
                          const device uint* inputs [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid>=6) return;
    uint offset=tid*6;
    if (tid<3) {
        threefry2x32_ctr_t c={{inputs[offset],inputs[offset+1]}};
        threefry2x32_key_t k={{inputs[offset+4],inputs[offset+5]}};
        threefry2x32_ctr_t r=threefry2x32_R(20,c,k);
        out[4*tid]=r.v[0]; out[4*tid+1]=r.v[1];
        out[4*tid+2]=0; out[4*tid+3]=0;
    } else {
        philox4x32_ctr_t c={{inputs[offset],inputs[offset+1],inputs[offset+2],inputs[offset+3]}};
        philox4x32_key_t k={{inputs[offset+4],inputs[offset+5]}};
        philox4x32_ctr_t r=philox4x32_R(10,c,k);
        for (uint j=0;j<4;++j) out[4*tid+j]=r.v[j];
    }
}

#define MMC_BATCH_KERNEL(Name,Batch,Core,Reduce) \
kernel void Name(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { \
    mmc_batch_body<Batch,Core,Reduce>(out,p,tid); }
MMC_BATCH_KERNEL(montmory_pair,2,0,false)
MMC_BATCH_KERNEL(montmory_quad,4,0,false)
MMC_BATCH_KERNEL(montmory_mulhi_pair,2,1,false)
MMC_BATCH_KERNEL(montmory_spec_pair,2,2,false)
MMC_BATCH_KERNEL(montmory_spec_dual,2,3,false)
MMC_BATCH_KERNEL(montmory_pair_reduce,2,0,true)
MMC_BATCH_KERNEL(montmory_quad_reduce,4,0,true)
MMC_BATCH_KERNEL(montmory_mulhi_pair_reduce,2,1,true)
MMC_BATCH_KERNEL(montmory_spec_pair_reduce,2,2,true)
MMC_BATCH_KERNEL(montmory_spec_dual_reduce,2,3,true)
#undef MMC_BATCH_KERNEL

#define MMC_REDUCE_KERNEL(Name,Engine) \
kernel void Name(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { \
    reference_diagnostic_body<Engine,true>(out,p,tid); }
MMC_REDUCE_KERNEL(threefry_bulk_reduce,1)
MMC_REDUCE_KERNEL(philox_bulk_reduce,2)
MMC_REDUCE_KERNEL(montmory_u32_tiled_reduce,6)
MMC_REDUCE_KERNEL(write_baseline_reduce,7)
#undef MMC_REDUCE_KERNEL
