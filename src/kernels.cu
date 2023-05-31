#include "../headers/kernels.h"
#include "../headers/common.h"
#include "../headers/reduction.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>


#define MAX_THREADS  256


namespace gpu {

  __global__ void conflictMatrixKernel(int *conflictMatrix, int *adjMatrix, int *chromosome, unsigned int n) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n && col < n) {
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

    void chooseAndReduce(int* d_odata, int* d_idata,unsigned int size) {
      int blocks, threads;
      getNumBlocksAndThreads(size, blocks, threads);
      dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);
      int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);
    
      switch (threads) {
          case 512:reduce<int, 512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 256:reduce<int, 256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 128:reduce<int, 128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 64:reduce<int, 64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 32:reduce<int, 32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 16:reduce<int, 16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 8:reduce<int, 8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 4:reduce<int, 4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 2:reduce<int, 2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 1:reduce<int, 1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
      } 
    }
    
    int fittest(const int *chromosome) {
      int penalty;
      size_t bytes = n*n * sizeof(int);
      int * h_penaltyMatix = (int*)malloc(bytes);
      int * d_adjMatrix;
      int * d_chromosome;
      int * d_conflictMatrix;
      int * d_result;

      cudaMalloc((void**)&d_adjMatrix, bytes);
      cudaMalloc((void**)&d_conflictMatrix, bytes);
      cudaMalloc((void**)&d_chromosome, n*sizeof(int));
      cudaMalloc((void**)&d_result, n*sizeof(int));
    
      cudaMemset(d_conflictMatrix, 0, bytes);
      cudaMemset(d_result, 0, bytes);
      cudaMemcpy(d_adjMatrix, adj, bytes, cudaMemcpyHostToDevice);
      cudaMemcpy(d_chromosome, adj, n*sizeof(int), cudaMemcpyHostToDevice);

      unsigned int blockThreads = (n + MAX_THREADS - 1) / MAX_THREADS;
      dim3 conflictsGridDim(blockThreads, blockThreads);
      conflictMatrixKernel<<<conflictsGridDim, MAX_THREADS>>>(d_conflictMatrix, d_adjMatrix, d_chromosome, n);
      chooseAndReduce(d_result, d_conflictMatrix, n*n);

      cudaMemcpy(h_penaltyMatix, d_result, bytes, cudaMemcpyDeviceToHost);
      penalty = h_penaltyMatix[0];

      cudaFree(d_conflictMatrix);
      cudaFree(d_adjMatrix);
      cudaFree(d_chromosome);
      cudaFree(d_result);
    
      return penalty;
    }



  __global__ void crossoverKernel(int* newFirst, int* newSecond, int* first, int* second, int a, int n) {
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
      int size = n * sizeof(int);

      // Allocate device memory
      int* devFirst;
      int* devSecond;
      int* devNewFirst;
      int* devNewSecond;
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

  void chooseAndReduceToMax(int* d_odata, int* d_idata,unsigned int size) {
      int blocks, threads;
      getNumBlocksAndThreads(size, blocks, threads);
      dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);
      int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);
    
      switch (threads) {
          case 512:reduceToMax<int, 512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 256:reduceToMax<int, 256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 128:reduceToMax<int, 128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 64:reduceToMax<int, 64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 32:reduceToMax<int, 32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 16:reduceToMax<int, 16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 8:reduceToMax<int, 8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 4:reduceToMax<int, 4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 2:reduceToMax<int, 2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
          case 1:reduceToMax<int, 1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;
      } 
    }

  //   int colorCount(std::vector<int>* chromosome) {
  //     int n = chromosome->size();
  //     int* d_data;
  //     cudaMalloc((void**)&d_data, n * sizeof(int));
  //     cudaMemcpy(d_data, chromosome->data(), n * sizeof(int), cudaMemcpyHostToDevice);
  //
  //     // Set up grid and block dimensions
  //     // unsigned int blockSize = 256;
  //     // unsigned int numBlocks = (n + blockSize - 1) / blockSize;
  //
  //     // Allocate GPU memory for intermediate and final results
  //     // int* d_intermediate;
  //     // cudaMalloc((void**)&d_intermediate, numBlocks * sizeof(int));
  //     int* d_result;
  //     cudaMalloc((void**)&d_result, sizeof(int));
  //     chooseAndReduceToMax(d_result, d_data, n*sizeof(int));
  //
  //     // Invoke the reduce kernel
  //     // reduceToMax<int, blockSize><<<numBlocks, blockSize>>>(d_data, d_intermediate, n);
  //
  //     // Invoke the reduce kernel again to obtain the final result
  //     // reduceToMax<int, blockSize><<<1, blockSize>>>(d_intermediate, d_result, numBlocks);
  //
  //     // Copy the result from GPU to CPU
  //     int result;
  //     cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
  //
  //     // Clean up GPU memory
  //     cudaFree(d_data);
  //     // cudaFree(d_intermediate);
  //     cudaFree(d_result);
  //
  //     return result;
  // }


  __global__ void countColorsKernel(int* chromosome, int* colors, int size) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;

      if (tid < size) {
          atomicAdd(&colors[chromosome[tid] - 1], 1);
      }
  }

  __global__ void swapColorsKernel(int* chromosome, int* swapTab, int* newChromosome, int size) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;

      if (tid < size) {
          int color = chromosome[tid] - 1;
          newChromosome[tid] = (swapTab[color] == -1) ? chromosome[tid] : swapTab[color] + 1;
      }
  }

  std::vector<int>* minimalizeColorsGPU(std::vector<int>* chromosome, int maxColors) {
        int size = chromosome->size();

        // Allocate GPU memory for chromosome, colors, swapTab, and newChromosome
        int* d_chromosome;
        cudaMalloc((void**)&d_chromosome, size * sizeof(int));
        int* d_colors;
        cudaMalloc((void**)&d_colors, maxColors * sizeof(int));
        int* d_swapTab;
        cudaMalloc((void**)&d_swapTab, maxColors * sizeof(int));
        int* d_newChromosome;
        cudaMalloc((void**)&d_newChromosome, size * sizeof(int));

        // Copy input chromosome from CPU to GPU
        cudaMemcpy(d_chromosome, chromosome->data(), size * sizeof(int), cudaMemcpyHostToDevice);

        // Set up grid and block dimensions for counting colors
        unsigned int blockSizeCount = 256;
        unsigned int numBlocksCount = (size + blockSizeCount - 1) / blockSizeCount;

        // Invoke the countColorsKernel
        countColorsKernel<<<numBlocksCount, blockSizeCount>>>(d_chromosome, d_colors, size);

        // Set up grid and block dimensions for swapping colors
        unsigned int blockSizeSwap = 256;
        unsigned int numBlocksSwap = (size + blockSizeSwap - 1) / blockSizeSwap;

        // Invoke the swapColorsKernel
        swapColorsKernel<<<numBlocksSwap, blockSizeSwap>>>(d_chromosome, d_swapTab, d_newChromosome, size);

        // Copy the results from GPU to CPU
        std::vector<int>* newChromosome = new std::vector<int>(size);
        cudaMemcpy(newChromosome->data(), d_newChromosome, size * sizeof(int), cudaMemcpyDeviceToHost);

        // Clean up GPU memory
        cudaFree(d_chromosome);
        cudaFree(d_colors);
        cudaFree(d_swapTab);
        cudaFree(d_newChromosome);

        return newChromosome;
    }


}
