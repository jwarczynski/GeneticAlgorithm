#include "../headers/benchmark.h"

#include "../headers/gpu.h"
#include "../headers/parallel.h"
#include "../headers/sequential.h"

#include "../headers/util.h"
#include "../headers/io.h"


void reportResult(geneticAlgorithm implementation, chromosome *samplePopulation) {
    auto start = chrono::steady_clock::now();
    ushort *coloring = (ushort*)malloc(n*sizeof(ushort));
    int result = implementation(samplePopulation, coloring);
    printf("%ld %d\n", since(start).count(), result);
    validateResult(coloring);
    saveColoringToFile(coloring, "5_gpu.txt");
}


void benchmarkResults() {
    chromosome *samplePopulation = generateSamplePopulation();
    geneticAlgorithm implementations[] = { seq::geneticAlg, parallel::geneticAlg, gpu::geneticAlg };
    int implemantationsNumber = sizeof(implementations) / sizeof(implementations[0]);
    
    for (int i=0; i<implemantationsNumber; ++i) {
      reportResult(implementations[i], samplePopulation);
    } 
}
