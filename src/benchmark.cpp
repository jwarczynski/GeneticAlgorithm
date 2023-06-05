#include "../headers/benchmark.h"

#include "../headers/gpu.h"
#include "../headers/parallel.h"
#include "../headers/sequential.h"
#include "../headers/util.h"
#include <iterator>
#include <ctime>
#include <cstdlib>
#include <sys/types.h>


void reportResult(geneticAlgorithm implementation, chromosome *samplePopulation) {
    auto start = chrono::steady_clock::now();
    ushort *coloring = (ushort*)malloc(n*sizeof(ushort));
    int result = implementation(samplePopulation, coloring);
    printf("%ld %d\n", since(start).count(), result);
    // cout << since(start).count() << " " << result << endl;
    validateResult(coloring);
}


void benchmarkResults() {
    chromosome *samplePopulation = generateSamplePopulation();
    geneticAlgorithm implementations[] = { seq::geneticAlg };
    // geneticAlgorithm implementations[] = { parallel::geneticAlg, seq::geneticAlg };
    // geneticAlgorithm implementations[] = {gpu::geneticAlg, parallel::geneticAlg, seq::geneticAlg};
    int implemantationsNumber = sizeof(implementations) / sizeof(implementations[0]);
    
    for (int i=0; i<implemantationsNumber; ++i) {
      printf("reported %d\n", i);
      reportResult(implementations[i], samplePopulation);
      printf("reported %d\n", i);
    } 
}
