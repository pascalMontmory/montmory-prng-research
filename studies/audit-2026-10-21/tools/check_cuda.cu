// Native CUDA check, prepared on Linux without nvcc; must run on CUDA hardware.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../reviewed-source/cuda/src/mmc_kernel.cu"
namespace cpu {
#include "../reviewed-source/cuda/src/mmc_core.c"
}

static void check(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0), "device properties");
    std::printf("device=%s\n", device.name);
    uint64_t compared = 0, guards = 0;
    const uint64_t seeds[] = {0, 1, 42, UINT64_MAX};
    const uint32_t configurations[][2] = {{1,1},{7,11},{256,3},{257,3},{8192,2}};
    for (uint64_t seed : seeds) for (auto& cfg : configurations) {
        uint32_t lanes = cfg[0], draws = cfg[1];
        uint64_t n = uint64_t(lanes) * draws;
        const uint64_t bases[] = {0,n,UINT64_MAX-3};
        for (uint64_t base : bases) for (unsigned block : {64u,128u,256u,512u}) {
            uint64_t capacity = ((uint64_t(lanes)+block-1)/block)*block*draws+16;
            uint64_t* output = nullptr;
            check(cudaMalloc(reinterpret_cast<void**>(&output), capacity*sizeof(uint64_t)), "allocate");
            check(cudaMemset(output, 0xff, capacity*sizeof(uint64_t)), "initialize guards");
            Params params{seed,base,draws,lanes};
            unsigned grid = unsigned((uint64_t(lanes)+block-1)/block);
            mmc_fill_kernel<<<grid,block>>>(output, params);
            check(cudaGetLastError(), "launch");
            check(cudaDeviceSynchronize(), "synchronize");
            std::vector<uint64_t> received(capacity);
            check(cudaMemcpy(received.data(), output, capacity*sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy");
            check(cudaFree(output), "free");
            for (uint32_t lane=0; lane<lanes; ++lane) {
                cpu::mmc_state_t state;
                cpu::mmc_init(&state,seed,lane);
                state.s += (base+uint64_t(lane)*draws)*state.gamma;
                for (uint32_t i=0; i<draws; ++i) {
                    uint64_t pos=uint64_t(lane)*draws+i;
                    uint64_t expected=cpu::mmc_next_u64(&state)>>11;
                    if (received[pos]!=expected) {
                        std::fprintf(stderr,"mismatch seed=%llu base=%llu position=%llu block=%u\n",
                          (unsigned long long)seed,(unsigned long long)base,(unsigned long long)pos,block);
                        return 1;
                    }
                    ++compared;
                }
            }
            for (uint64_t i=n; i<capacity; ++i) {
                if (received[i]!=UINT64_MAX) {
                    std::fprintf(stderr,"out-of-bounds write at %llu\n",(unsigned long long)i);
                    return 1;
                }
                ++guards;
            }
        }
    }
    std::printf("PASS: %llu exact mantissas; %llu guard words preserved\n",
                (unsigned long long)compared,(unsigned long long)guards);
}
