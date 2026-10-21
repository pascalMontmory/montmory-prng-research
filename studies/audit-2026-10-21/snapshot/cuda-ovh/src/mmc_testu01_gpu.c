#include "unif01.h"
#include "bbattery.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* API fournie par mmc_gpu_wrap.cu */
extern void   mmc_gpu_init(uint64_t seed, uint32_t threads, uint32_t draws_per_thread);
extern double mmc_gpu_next01(void);

/* Générateur externe pour TestU01 (U(0,1)) */
static double mmc_gpu_U01(void)
{
    return mmc_gpu_next01();
}


int main(int argc, char **argv)
{
    /* valeurs par défaut comme dans ton Makefile Metal */
    uint64_t seed    = 42;
    uint32_t threads = 2048;
    uint32_t draws   = 4096;
    const char *mode = "small";  /* small / crush / big */

    if (argc >= 4) {
        seed    = (uint64_t)strtoull(argv[1], NULL, 10);
        threads = (uint32_t)strtoul (argv[2], NULL, 10);
        draws   = (uint32_t)strtoul (argv[3], NULL, 10);
    }
    if (argc >= 5) {
        mode = argv[4];
    }

    printf("mmc_test_gpu: seed=%llu threads=%u draws=%u mode=%s\n",
           (unsigned long long)seed, (unsigned)threads, (unsigned)draws, mode);

    /* Init du générateur GPU (alloue le buffer et rempli le ring) */
    mmc_gpu_init(seed, threads, draws);

    /* Création d'un générateur TestU01 basé sur mmc_gpu_next01() */
    unif01_Gen *gen = unif01_CreateExternGen01("mmc_gpu", mmc_gpu_U01);

    /* Sélection de la batterie en fonction du mode */
    if (strcmp(mode, "small") == 0) {
        bbattery_SmallCrush(gen);
    } else if (strcmp(mode, "crush") == 0) {
        bbattery_Crush(gen);
    } else if (strcmp(mode, "big") == 0) {
        bbattery_BigCrush(gen);
    } else {
        fprintf(stderr, "Mode inconnu '%s' (utiliser: small, crush, big). SmallCrush par défaut.\n", mode);
        bbattery_SmallCrush(gen);
    }

    unif01_DeleteExternGen01(gen);
    return 0;
}
