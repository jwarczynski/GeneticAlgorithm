
#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include <vector>

#include "types.h"


__global__ void conflictMatrixKernel(ushort *conflictMatrix, ushort *adjMatrix, ushort *chromosome, ushort n); 
std::vector<std::vector<int>*> *crossover(std::vector<int> *first, std::vector<int> *second, int n);


#endif 
