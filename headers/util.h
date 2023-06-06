#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <vector>
#include <chrono>

#include "types.h"
using namespace std;


struct Node **graph();
void show();
int *greedy_coloring_list(struct Node **adj);
int *greedy_coloring_matrix();

template <
        class result_t   = std::chrono::milliseconds,
        class clock_t    = std::chrono::steady_clock,
        class duration_t = std::chrono::milliseconds
>
result_t since(std::chrono::time_point<clock_t, duration_t> const& start){
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

vector<int> *greedy_matrix_arbitrary_vertex(int u);

void validateInputParams(int argc);
void setInputParameters(char* argv[]);
void validateResult(ushort *res);

int *greedy_matrix_arbitrary_vertex(uint u);
chromosome* generateSamplePopulation();

#endif 
