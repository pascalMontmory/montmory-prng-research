SDK := $(shell xcrun --show-sdk-path)
CXX := /opt/homebrew/opt/llvm/bin/clang++
CC  := /usr/bin/cc

TESTU_INC := /opt/homebrew/include
TESTU_LIB := /opt/homebrew/lib

CXXFLAGS := -O3 -std=c++17 -isysroot $(SDK)
OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc -fobjc-abi-version=2
CFLAGS  := -O3 -isysroot $(SDK) -I$(TESTU_INC)

LDFLAGS := -framework Foundation -framework Metal \
           -I$(TESTU_INC) -L$(TESTU_LIB) \
           -ltestu01 -lprobdist -lmylib -lgmp -lm

.PHONY: all metallib gpu small crush clean build

all: small
build: ; mkdir -p build logs

metallib: metal/mmc_kernel.metal | build
	xcrun -sdk macosx metal -c metal/mmc_kernel.metal -o build/mmc_kernel.air
	xcrun -sdk macosx metallib build/mmc_kernel.air -o build/mmc_kernel.metallib

build/mmc_core.o: src/mmc_core.c | build
	$(CC) $(CFLAGS) -c $< -o $@

build/mmc_gpu_wrap.o: src/mmc_gpu_wrap.mm | build
	$(CXX) $(OBJCXXFLAGS) -c $< -o $@

build/mmc_testu01_gpu.o: src/mmc_testu01_gpu.c | build
	$(CC) $(CFLAGS) -c $< -o $@

gpu: metallib build/mmc_core.o build/mmc_gpu_wrap.o build/mmc_testu01_gpu.o
	$(CXX) $(OBJCXXFLAGS) \
	  build/mmc_core.o build/mmc_gpu_wrap.o build/mmc_testu01_gpu.o \
	  $(LDFLAGS) -o build/mmc_test_gpu

small: gpu
	./build/mmc_test_gpu 42 2048 4096        | tee logs/smallcrush_gpu.log

crush: gpu
	./build/mmc_test_gpu 42 4096 8192 crush  | tee logs/crush_gpu.log


big: gpu
	./build/mmc_test_gpu 42 8192 16384 big | tee logs/bigcrush_gpu.log

clean:
	rm -rf build/* logs/*.log
