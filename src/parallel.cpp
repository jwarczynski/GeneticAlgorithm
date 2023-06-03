#include "../headers/parallel.h"


namespace parallel {

	int fittest(const int *chromosome) {
			int penalty = 0;

			#pragma omp parallel for reduction(+:penalty)
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
			#pragma omp parallel for reduction(+:penalty)
			for (int i = 1; i < n; i++) {
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
			int maks = 0;
			#pragma omp parallel for reduction(max:maks)
			for (int i = 0; i < n; i++) {
					int tmp = 0;
					for (int j = 0; j < n; j++) {
							if (adj[i][j] == 1) {
									tmp++;
							}
					}
					maks = max(maks, tmp);
			}
			return maks;
	}

	vector<pair<vector<int> *, int> *> *generatePopulation(int maxDegree) {
			auto *res = new vector<pair<vector<int> *, int> *>(POPULATION_SIZE - SAMPLE_SIZE);
			#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < (POPULATION_SIZE - SAMPLE_SIZE); i++) {
					auto *tmp = new vector<int>(n);
					for (int j = 0; j < n; j++) {
							int a = rand() % maxDegree + 1;
							tmp->at(j) = a;
					}
					auto *p = new pair<vector<int> *, int>;
					*p = make_pair(tmp, fittest(tmp));
					res->at(i) = p;
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
			auto *newFirst = new vector<int>(n);
			auto *newSecond = new vector<int>(n);
			#pragma omp parallel
			{
				#pragma	omp for schedule(dynamic)
				for (int i = 0; i < a; i++) {
						newFirst->at(i) = (second->at(i));
						newSecond->at(i) = (first->at(i));
				}
				#pragma	omp for schedule(dynamic) nowait
				for (int i = a; i < n; i++) {
						newFirst->at(i) = first->at(i);
						newSecond->at(i) = second->at(i);
				}
			}
			auto *res = new vector<vector<int> *>;
			res->push_back(newFirst);
			res->push_back(newSecond);
			return res;
	}

  int **crossover(int *first, int *second) {
			int a = rand() % (n - 1);
      int *newFirst = (int*)malloc(n*sizeof(int));
      int *newSecond = (int*)malloc(n*sizeof(int));
			#pragma omp parallel
			{
				#pragma	omp for schedule(dynamic)
				for (int i = 0; i < a; i++) {
						newFirst[i] = second[i];
						newSecond[i] = first[i];
				}
				#pragma	omp for schedule(dynamic) nowait
				for (int i = a; i < n; i++) {
						newFirst[i] = first[i];
						newSecond[i] = second[i];
				}
			}
			int **res = (int**)malloc(2*sizeof(int*));
      res[0] = newFirst;
      res[1] = newSecond;
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
			#pragma omp parallel for reduction(max : res)
			for (int gene: *chromosome) {
					res = max(res, gene);
			}
			return res;
	}

	int colorCount(vector<pair<vector<int> *, int> *> *population) {
			int res = 0;
			#pragma omp parallel for reduction(max: res)
			for (pair<vector<int> *, int> *chromosome: *population) {
					res = max(res, colorCount(chromosome->first));
			}
			return res;
	}

	vector<int> *minimalizeColors(vector<int> *chromosome, int maxColors) {
			auto *newChromosome = new vector<int>(chromosome->size(), 0);
			vector<int> colors(maxColors);
			#pragma omp parallel for schedule(dynamic)
			for (vector<int>::size_type i = 0; i < chromosome->size(); ++i) {
					++colors[chromosome->at(i) - 1];
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
			#pragma omp parallel for schedule(dynamic)
			for (vector<int>::size_type i = 0; i < chromosome->size(); i++) {
					newChromosome->at(i) = swapTab.at(chromosome->at(i) - 1) + 1;
			}
			return newChromosome;
	}

	vector<int> *mate(vector<int> *mother, vector<int> *father, int maxColors) {
			auto res = new vector<int>(mother->size(), 0);
			auto toMutate = new vector<int>;
			#pragma omp parallel for
			for (vector<int>::size_type i = 0; i < mother->size(); i++) {
					int a = rand() % 100;
					if (a < 45) {
							(*res)[i] = mother->at(i);
					} else if (a < 90) {
							(*res)[i] = father->at(i);
					} else {
							(*res)[i] = -1;
							#pragma omp critical
							toMutate->push_back(i);
					}
			}
			for(auto gene: *toMutate){
					mutate(res, maxColors, gene);
			}
			return res;
	}

	vector<pair<vector<int> *, int> *> *newPopVol2(population_t *population, int maxColors) {
			auto *newPopulation = new population_t(POPULATION_SIZE, 0);
			#pragma omp parallel
			{
				#pragma omp for schedule(dynamic)
				for (int i = 0; i < POPULATION_SIZE / 10; i++) {
						newPopulation->at(i) = population->at(i);
				}
				#pragma omp for schedule(dynamic) nowait
				for (vector<int>::size_type i = population->size() / 10; i < population->size(); i++) {
						int mother = rand() % (POPULATION_SIZE / 2);
						int father = rand() % (POPULATION_SIZE / 2);
						while (father == mother) {
								father = (father + 1) % (POPULATION_SIZE / 2);
						}
						auto *p = new pair<vector<int> *, int>;
						*p = make_pair(mate(population->at(mother)->first, population->at(father)->first, maxColors), 0);
						p->second = fittest(p->first);
						newPopulation->at(i) = p;
				}
			}
			return newPopulation;
	}

	vector<pair<vector<int> *, int> *> *devaluate(vector<pair<vector<int> *, int> *> *population, int maxColors) {
			auto *newPopulation = new vector<pair<vector<int> *, int> *>(POPULATION_SIZE);
			#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < POPULATION_SIZE; i++) {
					pair<vector<int> *, int> *p = population->at(i);
					auto *newChromosome = new vector<int>;
					for (int gene: *p->first) {
							if (gene == maxColors) {
									newChromosome->push_back(gene - 1);
							} else {
									newChromosome->push_back(gene);
							}
					}
					auto *pr = new pair<vector<int> *, int>;
					*pr = make_pair(newChromosome, fittest(newChromosome));
					newPopulation->at(i) = pr;
			}
			return newPopulation;
	}

	int geneticAlg(population_t *sample, uint iterations, std::vector<int> *res) {
			int colors = 0;
			int mDeg;
			if (sample->empty()) {
					mDeg = maxDegree();
			} else {
					mDeg = colorCount(sample);
          *res = *sample->at(0)->first;
			}
			vector<pair<vector<int> *, int> *> *population;
			vector<pair<vector<int> *, int> *> *newPopulation;
			population = generatePopulation(mDeg-1);
			colors = colorCount(population);
			for (pair<vector<int> *, int> *s: *sample) {
					population->push_back(s);
			}
			sort(population->begin(), population->end(), comp);
			unsigned int t = 0;
			int best = mDeg;
			//while (since(start).count() < 300000) {
			while (t < iterations) {
					t++;
					newPopulation = newPopVol2(population, colors);
					population = newPopulation;
					colors = colorCount(population);
					for (auto &i: *population) {
							vector<int> *tmp = minimalizeColors(i->first, colors);
							i->first = tmp;
					}

					colors = colorCount(population);
					sort(population->begin(), population->end(), comp);
					if (population->at(0)->second == 0) {
							if(colors < best){
									best = colors;
                  *res = *population->at(0)->first;
							}
							population = devaluate(population, best-1);
							colors--;
					}
			}
			return best;
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

	vector<pair<vector<int> *, int> *> *generateSmallSample(){
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
