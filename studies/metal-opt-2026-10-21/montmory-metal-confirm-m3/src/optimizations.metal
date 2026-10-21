// Exact rewrites of the v3 Montmory generator. Append after generated/v3.metal.
template<bool Specialized>
inline uint opt_f(uint x,ulong key,uint ro,uint re) {
    if(!Specialized) return mmc_f_mulhi(x,key,ro,re);
    uint factor=1u+((x&1u)<<1);
    uint low=x*factor;
    uint high=metal::mulhi(x,factor);
    uint sum=low+uint(key);
    high+=uint(key>>32)+uint(sum<low);
    // Two rotations with seed-specific constants instead of a per-word shift.
    // The Metal compiler chooses whether to predicate or branch this expression.
    return (x&1u) ? (mmc_rot_low32(sum,high,ro)^(x>>1))
                  : (mmc_rot_low32(sum,high,re)^(x*5u));
}
template<bool Specialized>
inline void opt_round(thread MmcPair32& x,ulong key,uint ro,uint re) {
    uint next=x.hi^opt_f<Specialized>(x.lo,key,ro,re);
    x.hi=x.lo;x.lo=next;
}
inline MmcPair32 opt_add(MmcPair32 a,MmcPair32 b) {
    uint low=a.lo+b.lo;
    return MmcPair32{low,a.hi+b.hi+uint(low<a.lo)};
}
inline MmcPair32 opt_pair(ulong x) {return MmcPair32{uint(x),uint(x>>32)};}

template<uint Batch,bool Striped,bool Specialized,bool SplitCounter>
inline void opt_body(device ulong* out,constant BenchParams& p,uint tid) {
    MmcState state=mmc_batch_state<Specialized?2u:1u>(p);
    if(p.reserved==0xFFFFFFFFu) {
        for(uint i=tid;i<uint(p.words);i+=p.physical_threads) {
            uint lane=(i>>p.draws_shift)&(p.logical_lanes-1);
            ulong gamma=0x9E3779B97F4A7C15UL+2*ulong(lane);
            MmcPair32 x=opt_pair(p.seed+(p.base+ulong(i)+1)*gamma);
            opt_round<Specialized>(x,state.k0,state.r0,state.r1);
            opt_round<Specialized>(x,state.k1,state.r1,state.r2);
            opt_round<Specialized>(x,state.k2,state.r2,state.r0);
            out[i]=mmc_mix32(x)>>p.output_shift;
        }
        return;
    }
    uint width=1u<<p.reserved,lane=tid>>p.reserved;
    uint slot=tid&(width-1u);
    uint first=Striped?slot:slot*Batch;
    uint draws=1u<<p.draws_shift;
    if(lane>=p.logical_lanes || first>=draws)return;
    uint offset=lane<<p.draws_shift;
    uint spacing=Striped?width:1u;
    uint stride=width*Batch;
    ulong gamma=0x9E3779B97F4A7C15UL+2*ulong(lane);
    ulong counter=p.seed+(p.base+ulong(offset)+first+1)*gamma;
    ulong nextWord=gamma*ulong(spacing),nextBatch=gamma*ulong(stride);
    MmcPair32 counter32=opt_pair(counter),word32=opt_pair(nextWord),batch32=opt_pair(nextBatch);
    for(uint j=first;j<draws;j+=stride) {
        MmcPair32 x[Batch];
        if(SplitCounter) {
            MmcPair32 c=counter32;
#pragma unroll
            for(uint b=0;b<Batch;++b) {x[b]=c;c=opt_add(c,word32);}
        } else {
#pragma unroll
            for(uint b=0;b<Batch;++b)x[b]=opt_pair(counter+ulong(b)*nextWord);
        }
#pragma unroll
        for(uint b=0;b<Batch;++b)opt_round<Specialized>(x[b],state.k0,state.r0,state.r1);
#pragma unroll
        for(uint b=0;b<Batch;++b)opt_round<Specialized>(x[b],state.k1,state.r1,state.r2);
#pragma unroll
        for(uint b=0;b<Batch;++b)opt_round<Specialized>(x[b],state.k2,state.r2,state.r0);
#pragma unroll
        for(uint b=0;b<Batch;++b)if(j+b*spacing<draws)
            out[offset+j+b*spacing]=mmc_mix32(x[b])>>p.output_shift;
        if(SplitCounter)counter32=opt_add(counter32,batch32);else counter+=nextBatch;
    }
}
kernel void montmory_mulhi_quad(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<4,false,false,false>(out,p,tid);}
kernel void montmory_stripe_pair(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<2,true,false,false>(out,p,tid);}
kernel void montmory_stripe_quad(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<4,true,false,false>(out,p,tid);}
kernel void montmory_const_pair(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<2,false,true,false>(out,p,tid);}
kernel void montmory_const_stripe_pair(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<2,true,true,false>(out,p,tid);}
kernel void montmory_split_stripe_pair(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<2,true,false,true>(out,p,tid);}
kernel void montmory_const_stripe_quad(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<4,true,true,false>(out,p,tid);}
