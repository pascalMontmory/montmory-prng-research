#ifdef __cplusplus
extern "C" {
#include "unif01.h"
#include "bbattery.h"
}
#else
#include "unif01.h"
#include "bbattery.h"
#endif
#include <stdint.h>
#include <stdlib.h>  
extern double mmc_gpu_next01(void);
extern void   mmc_gpu_init(uint64_t seed, uint32_t threads, uint32_t draws_per_thread);
static double wrap(void){ return mmc_gpu_next01(); }
int main(int argc, char**argv){
  uint64_t seed=42; uint32_t threads=2048, dpt=4096;
  if (argc>1) seed    = strtoull(argv[1],0,10);
  if (argc>2) threads = (uint32_t)strtoul(argv[2],0,10);
  if (argc>3) dpt     = (uint32_t)strtoul(argv[3],0,10);
  mmc_gpu_init(seed, threads, dpt);
  unif01_Gen* gen = unif01_CreateExternGen01((char*)"Montmory-GPU-3R+DXSM", wrap);
  if (argc>4 && argv[4][0]=='c') bbattery_Crush(gen);
  else if (argc>4 && argv[4][0]=='b') bbattery_BigCrush(gen);
  else bbattery_SmallCrush(gen);
  unif01_DeleteExternGen01(gen);
  return 0;
}
