SRC_DIRS := src
INC_DIRS := headers
OBJ_DIR := bin
BIN_DIR := bin
CUDA_OBJ_DIR := $(BIN_DIR)/cuda
CPP_OBJ_DIR := $(BIN_DIR)/cpp

CPP_FILES := $(wildcard $(SRC_DIRS)/*.cpp)
CU_FILES := $(wildcard $(SRC_DIRS)/*.cu)

CPP_OBJS := $(patsubst $(SRC_DIRS)/%.cpp,$(CPP_OBJ_DIR)/%.o,$(CPP_FILES))
CU_OBJS := $(patsubst $(SRC_DIRS)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CU_FILES))

CXX := g++
NVCC := nvcc

CXXFLAGS := -std=c++14 -Wall -Wextra -fopenmp $(foreach D,$(INC_DIRS),-I$(D))
NVCCFLAGS := -std=c++14 $(foreach D,$(INC_DIRS),-I$(D))

EXECUTABLE := $(BIN_DIR)/main

all: $(EXECUTABLE)

$(EXECUTABLE): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) -lgomp $^ -o $@

$(CUDA_OBJ_DIR)/%.o: $(SRC_DIRS)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(CPP_OBJ_DIR)/%.o: $(SRC_DIRS)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(EXECUTABLE) $(CPP_OBJS) $(CU_OBJS)

