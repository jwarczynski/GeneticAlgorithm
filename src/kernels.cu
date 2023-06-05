#include "../headers/kernels.h"
#include "../headers/common.h"
#include "../headers/reduction.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_device_runtime_api.h>

#include "../headers/helper_cuda.h"

#include <iostream>
#include <sys/types.h>


#define MAX_THREADS  256
#define max(a, b) ((a) > (b) ? (a) : (b))

namespace gpu {

  __global__ void conflictMatrixKernel(ushort *conflictMatrix, ushort *adjMatrix, ushort *chromosome, ushort n) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n && col < n && col > row) {
      conflictMatrix[n*row + col] = (adjMatrix[n*row + col] == 1) && (chromosome[row] == chromosome[col]);
    }
    // threads = n * n-1
    // idx = tid / n + tid + 1
    // conflictMatrix[idx] = (adjMatrix[idx] == 1) && (chromosome[tid / n] == chromosome[tid%n +1]);
  }
  unsigned int nextPow2(unsigned int x) {
        --x;
      x |= x >> 1;
      x |= x >> 2;
      x |= x >> 4;
      x |= x >> 8;
      x |= x >> 16;
      return ++x;
    }

    void getNumBlocksAndThreads(int n, int &blocks, int &threads) {
      threads = (n < MAX_THREADS * 2) ? nextPow2((n + 1) / 2) : MAX_THREADS;
      blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    void chooseAndReduce(ushort* d_odata, ushort* d_idata, unsigned int size, int &blocks) {
      int threads;
      getNumBlocksAndThreads(size, blocks, threads);
      dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);
      int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);
    
      switch (threads) {
          case 512:reduce<ushort, 512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 256:reduce<ushort, 256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 128:reduce<ushort, 128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 64:reduce<ushort, 64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 32:reduce<ushort, 32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 16:reduce<ushort, 16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 8:reduce<ushort, 8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 4:reduce<ushort, 4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 2:reduce<ushort, 2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 1:reduce<ushort, 1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
      } 
    }
    
    ushort fittest(ushort *chromosome) {
      int penalty = 0;
      size_t bytes = n*n * sizeof(ushort);
      ushort * h_penaltyMatix = (ushort*)malloc(bytes);
      ushort * d_adjMatrix;
      ushort * d_chromosome;
      ushort * d_conflictMatrix;
      ushort * d_result;

      checkCudaErrors(cudaMalloc((void**)&d_adjMatrix, bytes));
      checkCudaErrors(cudaMalloc((void**)&d_conflictMatrix, bytes));
      checkCudaErrors(cudaMalloc((void**)&d_chromosome, n*sizeof(ushort)));
      checkCudaErrors(cudaMalloc((void**)&d_result, bytes));
    
      checkCudaErrors(cudaMemset(d_conflictMatrix, 0, bytes));
      checkCudaErrors(cudaMemset(d_result, 0, bytes));
      checkCudaErrors(cudaMemcpy(d_chromosome, chromosome, n*sizeof(ushort), cudaMemcpyHostToDevice));
      for (int i=0; i<n; ++i) {
        checkCudaErrors(cudaMemcpyAsync(d_adjMatrix + i*n, adj[i], n*sizeof(ushort), cudaMemcpyHostToDevice));
      }

      unsigned int blockThreads = (n + 32 - 1) / 32;
      dim3 conflictsGridDim(blockThreads, blockThreads, 1);
      dim3 conflictsThreadsDim(32, 32, 1);
      conflictMatrixKernel<<<conflictsGridDim, conflictsThreadsDim>>>(d_conflictMatrix, d_adjMatrix, d_chromosome, n);
      getLastCudaError("Kernel execution failed");

      int blocks;
      chooseAndReduce(d_result, d_conflictMatrix, n*n, blocks);
      getLastCudaError("Kernel execution failed");

      checkCudaErrors(cudaMemcpy(h_penaltyMatix, d_result, bytes, cudaMemcpyDeviceToHost));
      for (int i = 0; i < blocks; i++) {
        penalty += h_penaltyMatix[i];
      } 

      checkCudaErrors(cudaFree(d_conflictMatrix));
      checkCudaErrors(cudaFree(d_adjMatrix));
      checkCudaErrors(cudaFree(d_chromosome));
      checkCudaErrors(cudaFree(d_result));

      return penalty;
    }

  __global__ void crossoverKernel(ushort* newFirst, ushort* newSecond, ushort* first, ushort* second, ushort a, ushort n) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;

      if (tid < a) {
          newFirst[tid] = second[tid];
          newSecond[tid] = first[tid];
      } else if (tid < n) {
          newFirst[tid] = first[tid];
          newSecond[tid] = second[tid];
      }
  }

  std::vector<std::vector<int>*> *crossover(std::vector<int> *first, std::vector<int> *second, int n) {
      int a = rand() % (n - 1);
      int size = n * sizeof(ushort);

      // Allocate device memory
      ushort* devFirst;
      ushort* devSecond;
      ushort* devNewFirst;
      ushort* devNewSecond;
      cudaMalloc((void**)&devFirst, size);
      cudaMalloc((void**)&devSecond, size);
      cudaMalloc((void**)&devNewFirst, size);
      cudaMalloc((void**)&devNewSecond, size);

      // Copy input data to device memory
      cudaMemcpy(devFirst, first->data(), size, cudaMemcpyHostToDevice);
      cudaMemcpy(devSecond, second->data(), size, cudaMemcpyHostToDevice);

      // Launch kernel
      int blockSize = 256;
      int gridSize = (n + blockSize - 1) / blockSize;
      crossoverKernel<<<gridSize, blockSize>>>(devNewFirst, devNewSecond, devFirst, devSecond, a, n);

      // Copy result back to host memory
      std::vector<int>* newFirst = new std::vector<int>(n);
      std::vector<int>* newSecond = new std::vector<int>(n);
      cudaMemcpy(newFirst->data(), devNewFirst, size, cudaMemcpyDeviceToHost);
      cudaMemcpy(newSecond->data(), devNewSecond, size, cudaMemcpyDeviceToHost);

      // Free device memory
      cudaFree(devFirst);
      cudaFree(devSecond);
      cudaFree(devNewFirst);
      cudaFree(devNewSecond);

      // Create result vector
      std::vector<std::vector<int>*>* res = new std::vector<std::vector<int>*>();
      res->push_back(newFirst);
      res->push_back(newSecond);

      return res;
  }

  void chooseAndReduceToMax(ushort* d_odata, ushort* d_idata,unsigned int size, int &blocks) {
      int  threads;
      getNumBlocksAndThreads(size, blocks, threads);
      dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);
      int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);
    
      switch (threads) {
          case 512:reduceToMax<ushort, 512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 256:reduceToMax<ushort, 256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 128:reduceToMax<ushort, 128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 64:reduceToMax<ushort, 64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 32:reduceToMax<ushort, 32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 16:reduceToMax<ushort, 16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 8:reduceToMax<ushort, 8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 4:reduceToMax<ushort, 4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 2:reduceToMax<ushort, 2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 1:reduceToMax<ushort, 1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
      } 
    }

    ushort reduceToMax(ushort* input, ushort n) {
      // Device variables
      ushort *d_input, *d_output;
      ushort result;

      // Allocate memory on the device
      cudaMalloc((void**)&d_input, n * sizeof(ushort));
      cudaMalloc((void**)&d_output, n * sizeof(ushort));

      // Copy the input vector from the host to the device
      cudaMemcpy(d_input, input, n * sizeof(ushort), cudaMemcpyHostToDevice);

      // Determine the block and grid dimensions
      int blockSize = 256;
      int gridSize = (n + blockSize - 1) / blockSize;

      // Perform parallel reduction within each block
      reduceBlockMax<<<gridSize, blockSize, blockSize * sizeof(ushort)>>>(d_input, d_output, n);

      // Perform final reduction across block maximum values
      reduceFinalMax<<<1, blockSize>>>(d_output, gridSize);

      // Copy the final maximum value from the device to the host
      cudaMemcpy(&result, d_output, sizeof(ushort), cudaMemcpyDeviceToHost);

      // Free the allocated memory on the device
      cudaFree(d_input);
      cudaFree(d_output);

      return result;
    }

    ushort colorCount(ushort* chromosome) {
      return reduceToMax(chromosome, n); 
    }
  
  //   int colorCount(std::vector<int>* chromosome) {
  //     int bytes = n * sizeof(int);
  //     int* d_data;
  //     int* d_result;
  //     checkCudaErrors(cudaMalloc((void**)&d_data, bytes));
  //     checkCudaErrors(cudaMalloc((void**)&d_result, bytes));
  //     checkCudaErrors(cudaMemcpy(d_data, chromosome->data(), bytes, cudaMemcpyHostToDevice));
  //
  //     int blocks;
  //     chooseAndReduceToMax(d_result, d_data, n, blocks);
  //     getLastCudaError("Kernel execution failed");
  // 
  //     int result;
  //     int *partialMaxima = (int*)malloc(bytes);
  //     // checkCudaErrors(cudaMemcpy(partialMaxima, d_result, bytes, cudaMemcpyDeviceToHost));
  //     // for (int i=0; i< blocks; ++i) {
  //     //   result = max(result, partialMaxima[i]);
  //     // }
  //
  //     // Invoke the reduce kernel
  //     // reduceToMax<int, blockSize><<<numBlocks, blockSize>>>(d_data, d_intermediate, n);
  //
  //     // Invoke the reduce kernel again to obtain the final result
  //     // reduceToMax<int, blockSize><<<1, blockSize>>>(d_intermediate, d_result, numBlocks);
  //
  //     // Copy the result from GPU to CPU
  //
  //     // Clean up GPU memory
  //     checkCudaErrors(cudaFree(d_data));
  //     // cudaFree(d_intermediate);
  //     checkCudaErrors(cudaFree(d_result));
  //     free(partialMaxima);
  //
  //     return result;
  // }


  __global__ void countColorsKernel(ushort* chromosome, ushort* colors, int size) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;

      if (tid < size) {
          atomicAdd((int*)&colors[chromosome[tid] - 1], 1);
      }
  }

  __global__ void swapColorsKernel(ushort* chromosome, ushort* swapTab, ushort* newChromosome, int size) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;

      if (tid < size) {
          int color = chromosome[tid] - 1;
          newChromosome[tid] = (swapTab[color] == -1) ? chromosome[tid] : swapTab[color] + 1;
      }
  }

  // std::vector<int>* minimalizeColors(std::vector<int>* chromosome, int maxColors) {
  //       int* d_chromosome;
  //       int* d_colors;
  //       int* d_swapTab;
  //       int* d_newChromosome;

  //       checkCudaErrors(cudaMalloc((void**)&d_chromosome, n * sizeof(int)));
  //       checkCudaErrors(cudaMalloc((void**)&d_colors, maxColors * sizeof(int)));
  //       checkCudaErrors(cudaMalloc((void**)&d_swapTab, maxColors * sizeof(int)));
  //       checkCudaErrors(cudaMalloc((void**)&d_newChromosome, n * sizeof(int)));
  //
  //       checkCudaErrors(cudaMemcpy(d_chromosome, chromosome->data(), n * sizeof(int), cudaMemcpyHostToDevice));
  //       unsigned int blockSizeCount = 256;
  //       unsigned int numBlocksCount = (n + blockSizeCount - 1) / blockSizeCount;
  //
  //       countColorsKernel<<<numBlocksCount, blockSizeCount>>>(d_chromosome, d_colors, n);
  //       getLastCudaError("error invoking kernel");
  //
  //       // Set up grid and block dimensions for swapping colors
  //       unsigned int blockSizeSwap = 256;
  //       unsigned int numBlocksSwap = (n + blockSizeSwap - 1) / blockSizeSwap;
  //
  //       swapColorsKernel<<<numBlocksSwap, blockSizeSwap>>>(d_chromosome, d_swapTab, d_newChromosome, n);
  //       getLastCudaError("error invoking kernel");
  //
  //       std::vector<int>* newChromosome = new std::vector<int>(n);
  //       checkCudaErrors(cudaMemcpy(newChromosome->data(), d_newChromosome, n * sizeof(int), cudaMemcpyDeviceToHost));
  //
  //       checkCudaErrors(cudaFree(d_chromosome));
  //       checkCudaErrors(cudaFree(d_colors));
  //       checkCudaErrors(cudaFree(d_swapTab));
  //       checkCudaErrors(cudaFree(d_newChromosome));
  //
  //       return newChromosome;
  //   }


}
