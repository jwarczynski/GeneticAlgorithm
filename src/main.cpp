#include "../headers/gpu.h"
#include "../headers/parallel.h"
#include "../headers/sequential.h"


int n; // number of vertices in graph
int **adj; // matrix representing graph

int main(int argc, char* argv[]) {
		if (argc < 3) {
        cout << "Usage: program_name <file_name> <num_iterations>" << endl;
        return 1;
    }
    srand(time(NULL));
		string f_name = argv[1];
		read(f_name);
    auto *samplePopulation = generateSample();
    int max_color = 0;
    for (int i = 0; i < n; i++) {
        //cout << samplePopulation->at(0)->first->at(i) << "\t";
        max_color = max(max_color, samplePopulation->at(0)->first->at(i)+1);
    }
    unsigned int iterations = std::stoi(argv[2]);

    auto start = chrono::steady_clock::now();
    int gpu_result = gpu::geneticAlg(samplePopulation, iterations);
    cout << since(start).count() << " " << gpu_result << endl;
    
    start = chrono::steady_clock::now();
    int paralel_result = parallel::geneticAlg(samplePopulation, iterations);
    cout << since(start).count() << " " << paralel_result << endl;
    
    start = chrono::steady_clock::now();
    int seq_result = seq::geneticAlg(samplePopulation, iterations);
    cout << since(start).count() << " " << seq_result <<  endl;
}
