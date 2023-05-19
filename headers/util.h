#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <vector>
#include <chrono>
#include <queue>
using namespace std;

struct Node **graph();
void show();
void read(string name);
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
bool comp(pair<vector<int> *, int> *a, pair<vector<int> *, int> *b);
vector<pair<vector<int> *, int> *> *generateSample();
vector<int> *greedy_matrix_arbitrary_vertex(int u);

#endif 