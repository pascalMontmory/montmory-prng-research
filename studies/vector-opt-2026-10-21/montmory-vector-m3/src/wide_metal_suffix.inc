inline Wide64 key(ulong k) {return {splat(uint(k)),splat(uint(k>>32))};}
inline Wide64 counters(ulong c,ulong step) {
    return {V(uint(c),uint(c+step),uint(c+2*step),uint(c+3*step)),
            V(uint(c>>32),uint((c+step)>>32),uint((c+2*step)>>32),uint((c+3*step)>>32))};
}
template<bool Striped>
inline void body(device ulong* out,constant BenchParams& p,uint tid) {
    // Irregular worker counts keep the exact scalar indexing fallback.
    if(p.reserved==0xffffffffu) {
        MmcState s=mmc_prepared_state(p);
        for(uint i=tid;i<uint(p.words);i+=p.physical_threads) {
            uint lane=(i>>p.draws_shift)&(p.logical_lanes-1);
            ulong c=p.seed+(p.base+ulong(i)+1)*(0x9E3779B97F4A7C15UL+2*ulong(lane));
            out[i]=mmc_mix32(mmc_perm32(c,s))>>p.output_shift;
        }
        return;
    }
    uint width=1u<<p.reserved,lane=tid>>p.reserved,slot=tid&(width-1u);
    uint first=Striped?slot:4*slot,spacing=Striped?width:1u;
    uint draws=1u<<p.draws_shift;
    if(lane>=p.logical_lanes || first>=draws)return;
    uint offset=lane<<p.draws_shift,stride=4*width;
    ulong gamma=0x9E3779B97F4A7C15UL+2*ulong(lane);
    Wide64 c=counters(p.seed+(p.base+ulong(offset)+first+1)*gamma,ulong(spacing)*gamma);
    Wide64 step=key(ulong(stride)*gamma),k0=key(p.k0),k1=key(p.k1),k2=key(p.k2);
    for(uint j=first;j<draws;j+=stride) {
        Wide64 x=wide_round(c,k0,p.r0,p.r1);
        x=wide_round(x,k1,p.r1,p.r2);x=wide_round(x,k2,p.r2,p.r0);
        x=wide_mix(x);
#pragma unroll
        for(uint b=0;b<4;++b)if(j+b*spacing<draws)
            out[offset+j+b*spacing]=(ulong(x.lo[b])|(ulong(x.hi[b])<<32))>>p.output_shift;
        c=wide_add(c,step);
    }
}
}
kernel void montmory_vector4(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    montmory_wide_metal::body<false>(out,p,tid);
}
kernel void montmory_vector4_striped(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    montmory_wide_metal::body<true>(out,p,tid);
}
