#include "../headers/benchmark.h"
#include "../headers/util.h"

ushort n; // number of vertices in graph
ushort **adj; // matrix representing graph
unsigned int iterations;


int main(int argc, char* argv[]) {
    validateInputParams(argc);
    setInputParameters(argv);   
    
    srand(time(NULL));
    benchmarkResults();
}

