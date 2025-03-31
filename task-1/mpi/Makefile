MPICXX = mpic++
CXXFLAGS = -g -Wall

SRC = main.cpp
BIN_DIR = bin
TARGET = $(BIN_DIR)/main

NPROC = 5

build: $(BIN_DIR) $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(SRC)
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	mpiexec -n $(NPROC) $(TARGET)

clean:
	rm -rf $(BIN_DIR)

all: clean build run