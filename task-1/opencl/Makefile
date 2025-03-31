CC = g++
CFLAGS = -lOpenCL

SRC = main.cpp
BIN_DIR = bin
TARGET = $(BIN_DIR)/main

build: $(BIN_DIR) $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(SRC)
	$(CC) $(SRC) $(CFLAGS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BIN_DIR)

all: clean build run