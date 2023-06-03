#include <cstdio>

#include "../headers/common.h"
#include "../headers/util.h"


#include "../headers/gpu.h"
#include "../headers/parallel.h"
#include "../headers/sequential.h"

#include <queue>


//tworzy macierz sasiedztwa z pliku wejsciowego
struct Node **graph() {
    struct Node **arr = new struct Node *[n];
    struct Node *tmp;
    for (int i = 0; i < n; i++) {
        arr[i] = new struct Node;
        arr[i]->id = i;
        arr[i]->color = -1;
        arr[i]->child = nullptr;
        tmp = arr[i];
        for (int j = 0; j < n; j++) {
            if (adj[i][j] == 1) {
                tmp->child = new struct Node;
                tmp->child->id = j;
                tmp->child->color = -1;
                tmp->child->child = nullptr;
                tmp = tmp->child;
            }
        }
    }
    return arr;
}

void read(string name) {
    fstream f;
    f.open(name, ios::in);
    int a, b;
    f >> n;
    adj = new int *[n];
    for (int i = 0; i < n; i++) {
        adj[i] = new int[n];
        for (int j = 0; j < n; j++) {
            adj[i][j] = 0;
        }
    }
    while (!f.eof()) {
        f >> a >> b;
        a -= 1;
        b -= 1;
        adj[a][b] = 1;
        adj[b][a] = 1;
    }
    f.close();
}

//wyswietla macierz sasiedztwa
void show() {

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", adj[i][j]);
        }
        printf("\n");
    }
}

int *greedy_coloring_list(struct Node **adj) {
    struct Node *tmp;
    int cr;
    bool *available = new bool[n + 1];
    int *result = new int[n];
    for (int i = 0; i < n; i++) {
        available[i] = false;
        result[i] = -1;
    }

    result[0] = 1;
    cr = 1;

    for (int i = 1; i < n; i++) {
        tmp = adj[i]->child;
        while (tmp) {
            if (result[tmp->id] != -1)
                available[result[tmp->id]] = true;
            tmp = tmp->child;
        }

        for (cr = 1; cr <= n; cr++)
            if (!available[cr])
                break;


        result[i] = cr;
        for (int j = 0; j < n; j++)
            available[j] = false;
    }
    delete[] available;
    return result;
}

bool comp(pair<vector<int> *, int> *a, pair<vector<int> *, int> *b) {
    return a->second < b->second;
}

vector<int> *greedy_matrix_arbitrary_vertex(int u) {
	int color;
	auto *available = new vector<bool>;
	auto *result = new vector<int>;
	queue<int> q;
	for (int i = 0; i < n; i++) {
		available->push_back(false);
		result->push_back(-1);
	}
	available->push_back(false);
	q.push(u);
	result->at(u) = 1;
	while(!q.empty()) {
		while (!q.empty()) {
			u = q.front();
			q.pop();
			for (int j = 0; j < n; j++) {
				if (adj[u][j] == 1) {
					if (result->at(j) == -1) { q.push(j); }//wrzucam do kolejki nie przetworzone wierzcholki
					else {
						available->at(result->at(j)) = true;
					}
				}
			}
			for (color = 1; color <= n; color++) {
				if (!available->at(color)) {
					break;
				}// szukam najmniejszego koloru
			}
			result->at(u) = color;
			for (int j = 0; j < n; j++)
				available->at(j) = false;
		}
		for(int i = 0; i < n; i++){
			if(result->at(i) == -1){
				q.push(i);
				break;
			}
		}
	}
	return result;
}

int fittest(vector<int> *chromosome) {
			int penalty = 0;
			for (int i = 0; i < n; i++) {
					for (int j = i + 1; j < n; j++) {
							if (adj[i][j] == 1) {
									if (chromosome->at(i) == chromosome->at(j)) {
											penalty++;
									}
							}
					}
			}
			return penalty;
	}

int fittest(const int *chromosome) {
			int penalty = 0;
			for (int i = 0; i < n; i++) {
					for (int j = 0; j < n; j++) {
							if (adj[i][j] == 1) {
									if (chromosome[i] == chromosome[j]) {
											penalty++;
									}
							}
					}
			}
			return penalty;
	}

vector<pair<vector<int> *, int> *> *generateSample(){
			auto *samplePopulation = new vector<pair<vector<int> *, int> *>;
			for(int i = 0; i < SAMPLE_SIZE; i++){
					auto *sample = greedy_matrix_arbitrary_vertex(i);
					//cout << "Sample: " << i << endl;
					auto *samplePair = new pair<vector<int> *, int>;
					*samplePair = make_pair(sample, fittest(sample));
					samplePopulation->push_back(samplePair);
			}
			return samplePopulation;
	}

int *greedy_coloring_matrix() {
			int cr;
			bool *available = new bool[n+1];
			int *result = new int[n];
			for (int i = 0; i < n; i++) {
					available[i] = false;
					result[i] = -1;
			}

			result[0] = 1;
			cr = 1;

			for (int i = 1; i < n; i++) {
					for (int j = 0; j < n; j++) {
							if (result[j] != -1 && adj[i][j] == 1)
									available[result[j]] = true;
					}

					for (cr = 1; cr <= n; cr++)
							if (!available[cr])
									break;


					result[i] = cr;
					for (int j = 0; j < n; j++)
							available[j] = false;
			}
			delete[] available;
			return result;
}

void validateResult(std::vector<int> res) {
  for (int i=1;i<n;++i) {
    for (int j=i+1;j<n;j++) {
      if (adj[i][j] == 1 && res[i] == res[j]) {
        printf("vertex[%d](%d) is the same as vertex[%d](%d)\n", i, res[i], j, res[j]);
        return;
      }
    }
  }
}


  int calculateColorNum(population_t *population) {
     int max_color = 0;
     for (int i = 0; i < n; i++) {
        max_color = max(max_color, population->at(0)->first->at(i)+1);
      }
     return max_color;
  }

  void validateInputParams(int argc) {
    if (argc < 3) {
      cout << "Usage: program_name <file_name> <num_iterations>" << endl;
      exit(1);
    }
  }

  void setInputParameters(char* argv[]) {
    iterations = std::stoi(argv[2]);
		string f_name = argv[1];
		read(f_name);
  }

