#include "../headers/benchmark.h"

#include "../headers/gpu.h"
#include "../headers/parallel.h"
#include "../headers/sequential.h"


void reportResult(geneticAlgorithm implementation, population_t *samplePopulation) {
    auto start = chrono::steady_clock::now();
    vector<int> *coloring = new vector<int>(n);
    int result = implementation(samplePopulation, coloring);
    cout << since(start).count() << " " << result << endl;
    validateResult(*coloring);
}


void benchmarkResults() {
    auto *samplePopulation = generateSample();
    geneticAlgorithm implementations[] = {gpu::geneticAlg, parallel::geneticAlg, seq::geneticAlg};
    int implemantationsNumber = sizeof(implementations) / sizeof(implementations[0]);
    
    for (int i=0 ;i<implemantationsNumber ;++i) {
      reportResult(implementations[i], samplePopulation);
    } 
}
