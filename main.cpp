#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <ctime>
#include <algorithm>
#define POPULATION_SIZE 100
#define MUTATION_PERCENT 25

using namespace std;


struct Node {

    int id;
    int color;
    struct Node *child;
};

int n;
int **adj;

struct Node **graph() {

    struct Node **arr = new struct Node *[n];
    struct Node *tmp;
    for (int i = 0; i < n; i++) {
        arr[i] = new struct Node;
        arr[i]->id = i;
        arr[i]->color = -1;
        arr[i]->child = NULL;
        tmp = arr[i];
        for (int j = 0; j < n; j++) {
            if (adj[i][j] == 1) {
                tmp->child = new struct Node;
                tmp->child->id = j;
                tmp->child->color = -1;
                tmp->child->child = NULL;
                tmp = tmp->child;
            }
        }
    }
    return arr;
    /*for(int i = 0;i<n;i++)
     {
         tmp = arr[i];
         printf("\n %d-> ", i);
         while(tmp->child != NULL)
         {
             printf("%d ", tmp->child->id);
             tmp = tmp->child;
         }
     }
     wyswietla liste sasiedztwa
     */
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


//tworzy macierz sasiedztwa z pliku wejsciowego
void read(char *name) {

    FILE *f = fopen(name, "r");
    int a, b;
    fscanf(f, "%d", &n);
    adj = new int *[n];
    for (int i = 0; i < n; i++) {
        adj[i] = new int[n];
        for (int j = 0; j < n; j++) {
            adj[i][j] = 0;
        }
    }
    while (!feof(f)) {
        fscanf(f, "%d", &a);
        fscanf(f, "%d", &b);
        a -= 1;
        b -= 1;
        adj[a][b] = 1;
        adj[b][a] = 1;
    }
    fclose(f);
}

int *greedy_coloring_matrix() {
    int cr;
    bool *available = new bool[n];
    int *result = new int[n];
    for (int i = 0; i < n; i++) {
        available[i] = false;
        result[i] = -1;
    }

    result[0] = 0;
    cr = 0;

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (result[j] != -1 && adj[i][j] == 1)
                available[result[j]] = true;
        }

        for (cr = 0; cr < n; cr++)
            if (!available[cr])
                break;


        result[i] = cr+1;
        for (int j = 0; j < n; j++)
            available[j] = false;
    }
    delete[] available;
    /*for(int i=0;i<n;i++)
    {
        delete[] adj[i];
    }
    delete[] adj;*/
    return result;

}

int *greedy_coloring_list(struct Node **adj) {
    struct Node *tmp;
    int cr;
    bool *available = new bool[n];
    int *result = new int[n];
    for (int i = 0; i < n; i++) {
        available[i] = false;
        result[i] = -1;
    }

    result[0] = 0;
    cr = 0;

    for (int i = 1; i < n; i++) {
        tmp = adj[i]->child;
        while (tmp) {
            if (result[tmp->id] != -1)
                available[result[tmp->id]] = true;
            tmp = tmp->child;
        }

        for (cr = 0; cr < n; cr++)
            if (!available[cr])
                break;


        result[i] = cr;
        for (int j = 0; j < n; j++)
            available[j] = false;
    }
    delete[] available;
    /*for(int i=0;i<n;i++)
    {
        delete[] adj[i];
    }
    delete[] adj;*/
    return result;

}


//vector <int>

int fittest(const int *chromosome){
    int penalty = 0;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(adj[i][j] == 1){
                if(chromosome[i] == chromosome[j]){
                    penalty++;
                }
            }
        }
    }
    return penalty;
}

int fittest(vector <int> *chromosome){
    int penalty = 0;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(adj[i][j] == 1){
                if(chromosome->at(i) == chromosome->at(j)){
                    penalty++;
                }
            }
        }
    }
    return penalty;
}

int maxDegree(){
    int tmp = 0;
    int maks = 0;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(adj[i][j] == 1){
                tmp ++;
            }
        }
        maks = max(maks, tmp);
        tmp = 0;
    }
    return maks;
}
vector < vector <int>* > *generatePopulation (int maxDegree, int sampleSize){

    auto *res = new vector < vector < int >* >;
    for(int i = 0; i < POPULATION_SIZE-sampleSize; i++)
    {
        auto* tmp = new vector<int>;
        for(int j = 0; j < n; j++){
            int a = rand()%maxDegree+1;
            tmp->push_back(a);
        }
        res->push_back(tmp);
    }
    return res;
}
vector < vector < int >* > *newPop(vector < vector < int >* > *population)
{
    auto *newPopulation = new vector < vector < int >* >;
    for(int i = 0; i < 2; i++){
        shuffle(population->begin(),  population->end(), std::mt19937(std::random_device()()));
        vector < int > penalty;
        for (vector <int>* chromosome : *population){
            penalty.push_back(fittest(chromosome));
        }
        for(int j = 0; j < POPULATION_SIZE; j+=2){
            if(penalty[j] <= penalty[j + 1]){
                newPopulation->push_back(population->at(j));
            }else{
                newPopulation->push_back(population->at(j + 1));
            }
        }
    }
    return newPopulation;
}

vector <vector <int>* > *crossover(vector <int> *first, vector <int> *second){
    int a = rand()%(n-1);
    auto *newFirst = new vector<int>;
    auto *newSecond = new vector<int>;
    int i = 0;
    for(i; i < a; i++){
        newFirst->push_back(second->at(i));
        newSecond->push_back(first->at(i));
    }
    for(i; i < n; i++) {
        newFirst->push_back(first->at(i));
        newSecond->push_back(second->at(i));
    }
    auto *res = new vector <vector <int>* >;
    res->push_back(newFirst);
    res->push_back(newSecond);

    return res;
}

vector < vector < int >* > *newPopVol2(vector < vector < int >* > *population){
    auto *newPopulation = new vector < vector < int >* >;
    sort(population->begin(),  population->end());
    int i=0;
    for (i; i < population->size()/10;i++) {
        newPopulation->push_back(population->at(i));
    }
    for(i; i< population->size(); i++){
        int mother = rand()%(population->size()/2);
        int father = rand()%(population->size()/2);
        while(father==mother){
            father=(father+1)%(population->size()/2);
        }
        auto *tmp = crossover(population->at(mother), population->at(father));
        newPopulation->push_back(tmp->at(0));
    }
}


void mutate(vector <int> *chromosome, int maxColor){
    int a = rand()%n;
    vector<int> tabu;
    for(int i = 0; i < n; i++){
        if(adj[a][i] == 1){
            tabu.push_back(chromosome->at(i));
        }
    }
    int newColor = 0;
    while(find(tabu.begin(),  tabu.end(), newColor)==tabu.end()){
        newColor++;
    }
    chromosome->at(a) = newColor;
}

int colorCount(vector < vector < int >* > *population){
    int res = 0;
    for(const vector <int> *chromosome: *population){
        for(int gene: *chromosome){
            res = max(res, gene);
        }
    }
    return res;
}


vector <int> *minimalizeColors(vector <int> *chromosome, int maxColors){
    vector<int> colors(maxColors);
    for(int gene: *chromosome){
        colors.at(gene-1)++;
    }
    vector <int> swapTab(maxColors);
    int lowest = 0;
    for(int i = 0; i < colors.size(); i++){
        if(colors.at(i) == 0){
            swapTab.at(i) = -1;
        }
        else{
            swapTab.at(i) = lowest++;
        }
    }
    auto *newChromosome = new vector <int>;
    for(int i : *chromosome){
        newChromosome->push_back(swapTab.at(i-1)+1);
    }
    return newChromosome;
}

bool comp(vector<int>*a, vector<int>*b){
    return fittest(a) < fittest(b);
}

int geneticAlg(vector<vector<int>*> *sample){
    int colors = 0;
    int mDeg = maxDegree();
    vector<vector<int>* > *population;
    vector<vector<int>* > *newPopulation;
    population = generatePopulation(9, sample->size());
    for(vector <int> *s: *sample){
        population->push_back(s);
    }
    int t;
    while(fittest(population->at(0))> 0) {
        newPopulation = newPop(population);
        for (int i = 0; i < POPULATION_SIZE; i += 2) {
            auto *tmp = new vector<vector<int> *>;
            tmp = crossover(newPopulation->at(i), newPopulation->at(i + 1));
            newPopulation->at(i) = tmp->at(0);
            newPopulation->at(i + 1) = tmp->at(1);
        }
        population = newPopulation;
        colors = colorCount(population);
        for(int i = 0; i < population->size(); i++){
            vector<int> *tmp = minimalizeColors(population->at(i), colors);
            population->at(i) = tmp;
            if (rand() % 100 < MUTATION_PERCENT) {
                mutate(population->at(i), colors);
                mutate(population->at(i), colors);
                mutate(population->at(i), colors);
            }
        }
        colors = colorCount(population);
        sort(population->begin(),  population->end(), comp);
        int pen = fittest(population->at(0));
        cout << colors << "(" << pen << ")\t";
    }
    cout << "\nBest chromosome:\n\t";
    for(int gene : *population->at(0)){
        cout << gene << "\t";
    }
    cout << endl;
    cout << "Pen: "<< fittest(population->at(0));
    cout << endl;
    cout << "colors: " << colors << endl;
    return colors;
}



int main() {
    srand(time(NULL));
    char f_name[] = "gc30_50.txt";


    read(f_name);
//    show(matrix, n);
    int *outcome = greedy_coloring_matrix();
//    printf("\nwynik:\n");
//    for (int i = 0; i < n; i++) {
//        printf("wierzcholek %d --> %d\n", i + 1, outcome[i] + 1);
//    }

    auto *sample = new vector<int>;
    for(int i = 0; i < n; i++){
        sample->push_back(outcome[i]);
    }
    auto *samplePopulation = new vector<vector<int>*>;

    int max_color = 0;
    struct Node **lista;
    lista = graph();
    outcome = greedy_coloring_list(lista);




    FILE *output = fopen("output1000.txt", "w");
//    printf("\nwynik:\n");
//    for (int i = 0; i < n; i++) {
//        max_color = max(max_color, outcome[i] + 1);
//        fprintf(output, "wierzcholek %d --> %d\n", i + 1, outcome[i] + 1);
//        printf("wierzcholek %d --> %d\n", i + 1, outcome[i] + 1);
//    }
    for(int i=0; i< n ;i++ )
    {
        cout << outcome[i] << "\t";
        max_color = max(max_color, outcome[i] + 1);
    }
    cout << endl << "Penalty: " << fittest(outcome) << endl;
    printf("Uzyta ilosc kolorow: %d\n", max_color);
    cout << "Max Degree " << maxDegree() << endl;
    cout << "Final result: " << geneticAlg(samplePopulation);
}
