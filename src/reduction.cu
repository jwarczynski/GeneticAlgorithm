#include "../headers/reduction.h"

// #include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace gpu {
  __device__ ushort device_max(ushort a, ushort b) {
      return (a > b) ? a : b;
  }

  template <class T>
  struct SharedMemory {
    __device__ inline operator T *() {
      extern __shared__ int __smem[];
      return (T *)__smem;
    }

    __device__ inline operator const T *() const {
      extern __shared__ int __smem[];
      return (T *)__smem;
    }
  };

  template <class T, unsigned int blockSize>
  __global__ void reduce(T *g_idata, T *g_odata, unsigned int n) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n) mySum += g_idata[i + blockSize];

    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
      if (tid < s) {
        sdata[tid] = mySum = mySum + sdata[tid + s];
      }

      cg::sync(cta);
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32) {
      // Fetch final intermediate sum from 2nd warp
      if (blockSize >= 64) mySum += sdata[tid + 32];
      // Reduce final warp using shuffle
      for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        mySum += tile32.shfl_down(mySum, offset);
      }
    }

    // write result for this block to global mem
    if (cta.thread_rank() == 0) {
      g_odata[blockIdx.x] = mySum;
      // printf("mySum = %d\tglobal mem: %d\n", mySum, g_odata[blockIdx.x]);
    }
  }

  template <class T, unsigned int blockSize>
  __global__ void reduceToMax(T* g_idata, T* g_odata, unsigned int n) {
      // Handle to thread block group
      cg::thread_block cta = cg::this_thread_block();
      T *sdata = SharedMemory<T>();

      // Load data from global memory to shared memory
      unsigned int tid = threadIdx.x;
      unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

      T myMax = (i < n) ? g_idata[i] : 0;
      if (i + blockSize < n) {
          myMax = device_max(myMax, g_idata[i + blockSize]);
      }

      sdata[tid] = myMax;
      cg::sync(cta);

      // Perform reduction in shared memory
      for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
          if (tid < s) {
              myMax = device_max(myMax, sdata[tid + s]);
              sdata[tid] = myMax;
          }
          cg::sync(cta);
      }

      cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

      if (cta.thread_rank() < 32) {
          // Fetch final intermediate maximum from 2nd warp
          if (blockSize >= 64) {
              myMax = device_max(myMax, sdata[tid + 32]);
          }
          // Reduce final warp using shuffle
          for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
              myMax = device_max(myMax, tile32.shfl_down(myMax, offset));
          }
      }

      // Write result for this block to global memory
      if (cta.thread_rank() == 0) {
          g_odata[blockIdx.x] = myMax;
      }
  }

  __global__ void reduceBlockMax(ushort* input, ushort* output, ushort n) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements from global memory into shared memory
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = INT_MIN;
    }

    __syncthreads();

    // Perform parallel reduction using binary tree algorithm
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;

        if (index < blockDim.x) {
            sdata[index] = device_max(sdata[index], sdata[index + stride]);
        }

        __syncthreads();
    }

    // Store the maximum value in the first thread of each block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
  }

  // Kernel function for final reduction across block maximum values
  __global__ void reduceFinalMax(ushort* input, ushort n) {
      int tid = threadIdx.x;

      // Perform parallel reduction using binary tree algorithm
      for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
          if (tid < stride) {
              input[tid] = device_max(input[tid], input[tid + stride]);
          }

          __syncthreads();
      }
  }


  template __global__ void reduce<ushort, 512>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduce<ushort, 256>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduce<ushort, 128>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduce<ushort, 64>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduce<ushort, 32>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduce<ushort, 16>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduce<ushort, 8>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduce<ushort, 4>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduce<ushort, 2>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduce<ushort, 1>(ushort *g_idata, ushort *g_odata, unsigned int n);


  template __global__ void reduceToMax<ushort, 512>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduceToMax<ushort, 256>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduceToMax<ushort, 128>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduceToMax<ushort, 64>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduceToMax<ushort, 32>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduceToMax<ushort, 16>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduceToMax<ushort, 8>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduceToMax<ushort, 4>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduceToMax<ushort, 2>(ushort *g_idata, ushort *g_odata, unsigned int n);
  template __global__ void reduceToMax<ushort, 1>(ushort *g_idata, ushort *g_odata, unsigned int n);
}
