// Appended to the generated Metal arithmetic adapted to C++ by verify_emulation.py.
// The original Metal functions provide component references; complete streams
// are checked separately against the unchanged C reference in another library.
extern "C" unsigned component_checks(uint64_t* counts) {
    const uint32_t halves[]={0,1,2,3,0x7ffffffeu,0x7fffffffu,
                            0x80000000u,0x80000001u,0xfffffffeu,0xffffffffu};
    const uint64_t keys[]={0,1,UINT64_C(0xffffffff),UINT64_C(0x100000000),
        UINT64_C(0x100000001),UINT64_C(0x7fffffffffffffff),
        UINT64_C(0x8000000000000000),UINT64_MAX-1,UINT64_MAX};
    for(uint32_t r0=1;r0<64;r0+=2) for(uint32_t r1=1;r1<64;r1+=2)
    for(uint32_t x:halves) for(uint64_t k:keys) {
        if(mmc_f32(x,k,r0,r1)!=mmc_f(x,k,r0,r1)) return 1;
        ++counts[0];
        if(mmc_f_mulhi(x,k,r0,r1)!=mmc_f(x,k,r0,r1)) return 6;
        if(mmc_f_dual(x,k,r0,r1)!=mmc_f(x,k,r0,r1)) return 7;
        ++counts[5]; ++counts[6];
    }
    uint64_t random=42;
    for(unsigned i=0;i<(1u<<20);++i) {
        uint64_t x=random=splitmix64(random);
        uint64_t k=random=splitmix64(random);
        unsigned r0=unsigned(k&63)|1u,r1=unsigned((k>>8)&63)|1u;
        if(mmc_f32(uint32_t(x),k,r0,r1)!=mmc_f(uint32_t(x),k,r0,r1)) return 1;
        ++counts[0];
        if(mmc_f_mulhi(uint32_t(x),k,r0,r1)!=mmc_f(uint32_t(x),k,r0,r1)) return 6;
        if(mmc_f_dual(uint32_t(x),k,r0,r1)!=mmc_f(uint32_t(x),k,r0,r1)) return 7;
        ++counts[5]; ++counts[6];
        MmcPair32 pair={uint32_t(x),uint32_t(x>>32)};
        auto product=mmc_mul32(pair,uint32_t(k),uint32_t(k>>32));
        if((uint64_t(product.lo)|(uint64_t(product.hi)<<32))!=x*k) return 2;
        ++counts[1];
        for(unsigned shift:{25u,28u,27u}) {
            auto result=mmc_xor_right32(pair,shift);
            if((uint64_t(result.lo)|(uint64_t(result.hi)<<32))!=(x^(x>>shift))) return 3;
            ++counts[2];
        }
        // Cycle through all 32 possible final rotations, including zero.
        x=(x&((UINT64_C(1)<<59)-1))|(uint64_t(i&31)<<59);
        if(mmc_mix32(MmcPair32{uint32_t(x),uint32_t(x>>32)})!=pcg_dxsm(x)) return 4;
        ++counts[3];
        MmcState state; mmc_init(state,k,0);
        auto result=mmc_perm32(x,state);
        if((uint64_t(result.lo)|(uint64_t(result.hi)<<32))!=perm3(x,state)) return 5;
        ++counts[4];
    }
    for(uint64_t x:keys) {
        if(mmc_mix32(MmcPair32{uint32_t(x),uint32_t(x>>32)})!=pcg_dxsm(x)) return 4;
        ++counts[3];
        for(uint64_t k:keys) {
            auto product=mmc_mul32(MmcPair32{uint32_t(x),uint32_t(x>>32)},uint32_t(k),uint32_t(k>>32));
            if((uint64_t(product.lo)|(uint64_t(product.hi)<<32))!=x*k) return 2;
            ++counts[1];
        }
    }
    return 0;
}
