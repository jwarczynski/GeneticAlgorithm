#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <cuda_runtime.h>
#include <sys/types.h>

namespace gpu {

  template <class T, unsigned int blockSize>
  __global__ void reduce(T *g_idata, T *g_odata, ushort n);

  template <class T, unsigned int blockSize>
  __global__ void reduceToMax(T* g_idata, T* g_odata, ushort n);

  __global__ void reduceBlockMax(ushort* input, ushort* output, int n);
  __global__ void reduceFinalMax(ushort* input, ushort n);
}

#endif 
