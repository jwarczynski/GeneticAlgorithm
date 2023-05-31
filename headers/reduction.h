#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <cuda_runtime.h>

namespace gpu {

  template <class T, unsigned int blockSize>
  __global__ void reduce(T *g_idata, T *g_odata, unsigned int n);

  template <class T, unsigned int blockSize>
  __global__ void reduceToMax(T* g_idata, T* g_odata, unsigned int n);

}

#endif 
