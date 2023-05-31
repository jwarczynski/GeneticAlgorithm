
#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include "common.h"


__global__ void conflictMatrixKernel(int *conflictMatrix, int *adjMatrix, int *chromosome, unsigned int n); 
std::vector<std::vector<int>*> *crossover(std::vector<int> *first, std::vector<int> *second, int n);


#endif 
