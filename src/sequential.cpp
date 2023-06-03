#include "../headers/common.h"
#include "../headers/util.h"
#include "../headers/geneticAlgorithm.h"


using namespace std;

namespace seq {
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

	int maxDegree() {
			int tmp = 0;
			int maks = 0;
			for (int i = 0; i < n; i++) {
					for (int j = 0; j < n; j++) {
							if (adj[i][j] == 1) {
									tmp++;
							}
					}
					maks = max(maks, tmp);
					tmp = 0;
			}
			return maks;
	}

	vector<pair<vector<int> *, int> *> *generatePopulation(int maxDegree) {

			auto *res = new vector<pair<vector<int> *, int> *>;
			for (int i = 0; i < (POPULATION_SIZE - SAMPLE_SIZE); i++) {
					auto *tmp = new vector<int>;
					for (int j = 0; j < n; j++) {
							int a = rand() % maxDegree + 1;
							tmp->push_back(a);
					}
					auto *p = new pair<vector<int> *, int>;
					*p = make_pair(tmp, fittest(tmp));
					res->push_back(p);
			}
			return res;
	}

	vector<vector<int> *> *newPop(vector<vector<int> *> *population) {
			auto *newPopulation = new vector<vector<int> *>;
			for (int i = 0; i < 2; i++) {
					shuffle(population->begin(), population->end(), std::mt19937(std::random_device()()));
					vector<int> penalty;
					for (vector<int> *chromosome: *population) {
							penalty.push_back(fittest(chromosome));
					}
					for (int j = 0; j < POPULATION_SIZE; j += 2) {
							if (penalty[j] <= penalty[j + 1]) {
									newPopulation->push_back(population->at(j));
							} else {
									newPopulation->push_back(population->at(j + 1));
							}
					}
			}
			return newPopulation;
	}

	vector<vector<int> *> *crossover(vector<int> *first, vector<int> *second) {
			int a = rand() % (n - 1);
			auto *newFirst = new vector<int>;
			auto *newSecond = new vector<int>;
			int i = 0;
			for (; i < a; i++) {
					newFirst->push_back(second->at(i));
					newSecond->push_back(first->at(i));
			}
			for (; i < n; i++) {
					newFirst->push_back(first->at(i));
					newSecond->push_back(second->at(i));
			}
			auto *res = new vector<vector<int> *>;
			res->push_back(newFirst);
			res->push_back(newSecond);

			return res;
	}

	void mutate(vector<int> *chromosome, int maxColor, int a) {
			vector<int> tabu;
			for (int i = 0; i < n; i++) {
					if (adj[a][i] == 1) {
							tabu.push_back(chromosome->at(i));
					}
			}
			int newColor = 1;
			while (find(tabu.begin(), tabu.end(), newColor) != tabu.end()) {
					newColor++;
			}
			if(newColor >= maxColor){
					newColor = rand()%(maxColor-1)+1;
			}
			chromosome->at(a) = newColor;
	}

	int colorCount(vector<int> *chromosome) {
			int res = 0;
			for (int gene: *chromosome) {
					res = max(res, gene);
			}
			return res;
	}

	int colorCount(vector<pair<vector<int> *, int> *> *population) {
			int res = 0;
			for (pair<vector<int> *, int> *chromosome: *population) {
					res = max(res, colorCount(chromosome->first));
			}
			return res;
	}

	vector<int> *minimalizeColors(vector<int> *chromosome, int maxColors) {
			vector<int> colors(maxColors);
			for (int gene: *chromosome) {
					colors.at(gene - 1)++;
			}
			vector<int> swapTab(maxColors);
			int lowest = 0;
			for (vector<int>::size_type i = 0; i < colors.size(); i++) {
					if (colors.at(i) == 0) {
							swapTab.at(i) = 0;
					} else {
							swapTab.at(i) = lowest++;
					}
			}
			auto *newChromosome = new vector<int>;
			for (int i: *chromosome) {
					newChromosome->push_back(swapTab.at(i - 1) + 1);
			}
			return newChromosome;
	}

	vector<int> *mate(vector<int> *mother, vector<int> *father, int maxColors) {
			auto res = new vector<int>;
			auto toMutate = new vector<int>;
			for (vector<int>::size_type i = 0; i < mother->size(); i++) {
					int a = rand() % 100;
					if (a < 45) {
							res->push_back(mother->at(i));
					} else if (a < 90) {
							res->push_back(father->at(i));
					} else {
							res ->push_back(-1);
							toMutate->push_back(i);
					}
			}
			for(auto gene: *toMutate){
					mutate(res, maxColors, gene);
			}
			return res;
	}

	vector<pair<vector<int> *, int> *> *newPopVol2(vector<pair<vector<int> *, int> *> *population, int maxColors) {
			auto *newPopulation = new vector<pair<vector<int> *, int> *>;
			vector<int>::size_type i = 0;
			for (; i < population->size() / 10; i++) {
					newPopulation->push_back(population->at(i));
			}
			for (; i < population->size(); i++) {
					int mother = rand() % (population->size() / 2);
					int father = rand() % (population->size() / 2);
					while (father == mother) {
							father = (father + 1) % (population->size() / 2);
					}
					auto *p = new pair<vector<int> *, int>;
					*p = make_pair(mate(population->at(mother)->first, population->at(father)->first, maxColors), 0);
					p->second = fittest(p->first);
					newPopulation->push_back(p);
			}
			return newPopulation;
	}

	vector<pair<vector<int> *, int> *> *devaluate(vector<pair<vector<int> *, int> *> *population, int maxColors) {
			auto *newPopulation = new vector<pair<vector<int> *, int> *>;
			for (pair<vector<int> *, int> *p: *population) {
					auto *newChromosome = new vector<int>;
					for (int gene: *p->first) {
							if (gene == maxColors - 1) {
									newChromosome->push_back(gene);
							} else {
									newChromosome->push_back(gene);
							}
					}
					auto *pr = new pair<vector<int> *, int>;
					*pr = make_pair(newChromosome, fittest(newChromosome));
					newPopulation->push_back(pr);
			}
			return newPopulation;
	}

  int geneticAlg(population_t *sample, std::vector<int> *res) {
      helperFunctionsImpl implementations = {
        seq::maxDegree,
        seq::colorCount,
        seq::generatePopulation,
        seq::newPopVol2,
        seq::minimalizeColors,
        seq::devaluate
      };

      return geneticAlg(sample ,res, implementations);
    }

	void translate(string name) {
			fstream input;
			fstream output;
			string buffer;
			input.open(name+".col.b", ios::in|ios::binary);
			output.open(name+".txt", ios::out);
			while(!input.eof()){
					getline(input, buffer, '\n');
					output << buffer << endl;
			}
			input.close();
			output.close();
	}

	vector<pair<vector<int> *, int> *> *generateSmallSample() {
			auto *samplePopulation = new vector<pair<vector<int> *, int> *>;
			auto *sample = greedy_coloring_matrix();
			auto *newSample = new vector<int>;
			for(int i = 0; i < n; i++){
					newSample->push_back(sample[i]);
			}
			auto *samplePair = new pair<vector<int> *, int>;
			*samplePair = make_pair(newSample, fittest(newSample));
			samplePopulation->push_back(samplePair);
			return samplePopulation;
	}
}
