# Zmienna z nazwą folderu, w którym znajdują się pliki źródłowe
SRC_DIR := src

# Zmienna z nazwą folderu, w którym ma zostać umieszczony plik wykonywalny
BIN_DIR := bin

# Lista plików źródłowych (rozszerzenie .cpp)
SRCS := $(wildcard $(SRC_DIR)/*.cpp)

# Nazwa pliku wykonywalnego
EXECUTABLE := $(BIN_DIR)/main

# Kompilator C++
CXX := g++

# Opcje kompilacji
CXXFLAGS := -std=c++14 -Wall -Wextra -fopenmp -Iheaders

all: $(EXECUTABLE)

$(EXECUTABLE): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE) $(ARGS)

clean:
	rm -f $(EXECUTABLE)

