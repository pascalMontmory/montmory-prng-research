#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include "mmc_core.h"

int main(int argc, char**argv){
  int lanes = (argc>1) ? atoi(argv[1]) : omp_get_max_threads();
  int per_lane = (argc>2) ? atoi(argv[2]) : 5*1000*1000;  /* 5M tirages/lane */
  uint64_t seed = 42;

  double sum = 0.0;
  #pragma omp parallel reduction(+:sum)
  {
    int lane = omp_get_thread_num();
    mmc_state_t st; mmc_init(&st, seed, (uint64_t)lane);
    double local=0.0;
    for(int i=0;i<per_lane;i++){
      double u = mmc_next01(&st);
      local += u;                 /* exemple: somme simple */
    }
    sum += local;
  }
  double mean = sum/( (double)lanes*per_lane );
  printf("lanes=%d per_lane=%d -> mean=%.6f (expected 0.5)\n", lanes, per_lane, mean);
  return 0;
}
