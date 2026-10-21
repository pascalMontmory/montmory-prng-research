#ifndef MONTMORY_LANE_REFERENCE_HPP
#define MONTMORY_LANE_REFERENCE_HPP
#include "reference.hpp"

// Efficient independent oracle: the recovered scalar C core, lane by lane.
// Random123 consumes every word from each generated block.
static inline std::vector<uint64_t> laneReference(const BenchParams& p,unsigned engine) {
    std::vector<uint64_t> values(p.words);
    if(isMontmory(engine)) {
        const uint64_t draws=UINT64_C(1)<<p.draws_shift;
        for(uint32_t lane=0;lane<p.logical_lanes;++lane) {
            mmc_state_t st;mmc_init(&st,p.seed,lane);
            const uint64_t offset=uint64_t(lane)*draws;
            st.s+=(p.base+offset)*st.gamma;
            for(uint64_t j=0;j<draws;++j) values[offset+j]=mmc_next_u64(&st)>>p.output_shift;
        }
    } else if(engine==2) {
        philox4x32_key_t key={{p.key0,p.key1}};
        for(uint64_t i=0;i<p.words/2;++i) {
            const uint64_t index=p.base+i;
            philox4x32_ctr_t ctr={{uint32_t(index),uint32_t(index>>32),0,0}};
            auto v=philox4x32_R(10,ctr,key);
            values[2*i]=(uint64_t(v.v[0])|(uint64_t(v.v[1])<<32))>>p.output_shift;
            values[2*i+1]=(uint64_t(v.v[2])|(uint64_t(v.v[3])<<32))>>p.output_shift;
        }
    } else {
        for(uint64_t i=0;i<p.words;++i) values[i]=referenceWord(engine,p,i);
    }
    return values;
}
#endif
