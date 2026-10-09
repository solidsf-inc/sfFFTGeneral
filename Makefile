NVCC ?= nvcc
ARCHS ?= 90 100 103 121
NVCCFLAGS ?= -O3 -std=c++17 --threads 1
BUILD_DIR ?= build
GENCODE := $(foreach arch,$(ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch)) -gencode arch=compute_90,code=compute_90

all: $(BUILD_DIR)/check

$(BUILD_DIR)/check: tests/check.cu src/kernel.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $< -o $@

test: all
	$(BUILD_DIR)/check

.PHONY: all test
