.PHONY: all clear run debug
.DEFAULT_GOAL := all

### directory structure ###
bin: ; mkdir -p bin bin/kernels

### C compilation ###
C_SRC_FILES := $(shell find src/host -name *.c)
C_FLAGS := -lcuda
bin/main: $(C_SRC_FILES) bin ; $(CC) $(C_FLAGS) -g $(C_SRC_FILES) -o $@


### CUDA compilation ###
NVCC := nvcc
CUDA_SRC_FILES := $(shell find src/device -name *.cu)
CUDA_KERNELS := $(patsubst src/device/%.cu, bin/kernels/%.cubin, $(CUDA_SRC_FILES))
CUDA_FLAGS := -cubin -gencode arch=compute_30,code=sm_30
bin/kernels/%.cubin : src/device/%.cu bin ; $(NVCC) $(CUDA_FLAGS) $< -o $@

### default build target ###
all: bin/main $(CUDA_KERNELS)

### runs the executable ###
run: ; @if [ -e ./bin/main ]; then cd ./bin && ./main; else echo "No build ready"; fi

### gdb ###
debug: ; @if [ -e ./bin/main ]; then cd ./bin && gdb ./main; else echo "No build ready"; fi

### clears build files ###
clear: ; @rm -r bin bin/* 2>/dev/null ; echo "Directory cleared"
