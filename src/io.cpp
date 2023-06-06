#include "../headers/io.h"

#include <algorithm>
#include <fstream>
#include <iostream>

void saveColoringToFile(ushort *coloring, const std::string& filename) {
  std::string folder = "Graph-Visualizer/solutions/"; 
  std::ofstream file(folder + filename); 

  if (file.is_open()) {
    ushort maxColor = *std::max_element(coloring, coloring + n);
    file << maxColor << std::endl;
    
    for (int i = 0; i < n; i++) {
      file << coloring[i] << std::endl;
    }

    file.close();
  } else {
    std::cout << "Nie można otworzyć pliku" << std::endl;
  }
}

void read(std::string name) {
  std::fstream f;
  f.open(name, std::ios::in);
  int a, b;
  f >> n;
  adj = new ushort *[n];
  for (int i = 0; i < n; i++) {
    adj[i] = new ushort[n];
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

void translate(std::string name) {
    std::fstream input;
    std::fstream output;
    std::string buffer;
    input.open(name+".col.b", std::ios::in|std::ios::binary);
    output.open(name+".txt", std::ios::out);
    while(!input.eof()){
            getline(input, buffer, '\n');
            output << buffer << std::endl;
    }
    input.close();
    output.close();
}
