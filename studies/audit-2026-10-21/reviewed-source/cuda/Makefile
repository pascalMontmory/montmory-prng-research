# Makefile CUDA pour mmc_test_gpu avec TestU01 (Linux / OVH GPU)
#
# Structure attendue :
#   src/mmc_core.c           - PRNG core (CPU, pour TestU01)
#   src/mmc_kernel.cu        - kernel CUDA (port de mmc_kernel.metal)
#   src/mmc_gpu_wrap.cu      - wrapper CUDA (port de mmc_gpu_wrap.mm)
#   src/mmc_testu01_gpu.c    - harness TestU01 qui appelle mmc_gpu_next01()
#
# Targets :
#   make small   -> SmallCrush sur GPU
#   make crush   -> Crush sur GPU
#   make big     -> BigCrush sur GPU
#   make gpu     -> compile seulement l’exécutable GPU TestU01
#   make clean

NVCC      := nvcc
CC        := gcc

# Adapte ces chemins à ton installation TestU01
TESTU_INC := /usr/include/testu01
TESTU_LIB := /usr/lib/x86_64-linux-gnu

CFLAGS  := -O3 -I$(TESTU_INC)

NVFLAGS := -O3 -std=c++17 -I$(TESTU_INC)

LDFLAGS := -L$(TESTU_LIB) -ltestu01 -lprobdist -lmylib -lgmp -lm




SRC_CORE      := src/mmc_core.c
SRC_KERNEL_CU := src/mmc_kernel.cu
SRC_WRAP_CU   := src/mmc_gpu_wrap.cu
SRC_TU01      := src/mmc_testu01_gpu.c

OBJ_CORE      := build/mmc_core.o
OBJ_KERNEL_CU := build/mmc_kernel.o
OBJ_WRAP_CU   := build/mmc_gpu_wrap.o
OBJ_TU01      := build/mmc_testu01_gpu.o

BIN           := build/mmc_test_gpu

.PHONY: all gpu small crush big clean build

all: small

build:
	mkdir -p build logs

# --- Compilation ---

$(OBJ_CORE): $(SRC_CORE) | build
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_KERNEL_CU): $(SRC_KERNEL_CU) src/mmc_cuda_params.h | build
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(OBJ_WRAP_CU): $(SRC_WRAP_CU) src/mmc_cuda_params.h | build
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(OBJ_TU01): $(SRC_TU01) | build
	$(CC) $(CFLAGS) -c $< -o $@

# --- Link GPU TestU01 binary ---

gpu: $(OBJ_CORE) $(OBJ_KERNEL_CU) $(OBJ_WRAP_CU) $(OBJ_TU01)
	$(NVCC) $(NVFLAGS) $^ $(LDFLAGS) -o $(BIN)

# --- Batteries TestU01 (même interface que sur Mac) ---

small: gpu
	./$(BIN) 42 2048 4096        | tee logs/smallcrush_gpu.log

crush: gpu
	./$(BIN) 42 4096 8192 crush  | tee logs/crush_gpu.log

big: gpu
	./$(BIN) 42 8192 16384 big   | tee logs/bigcrush_gpu.log

clean:
	rm -rf build logs
