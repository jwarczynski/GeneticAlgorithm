#include "../headers/util.h"

int n; // number of vertices in graph
int **adj; // matrix representing graph
unsigned int iterations;


int main(int argc, char* argv[]) {
		validateInputParams(argc);
    setInputParameters(argv);   
    
    srand(time(NULL));
    benchmarkResults();
}

