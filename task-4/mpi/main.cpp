#include <mpi.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstdlib>

#define N 2000

MPI_Status status;

double matrixA[N][N], matrixB[N][N], matrixC[N][N];

void generateMatrix(int rows, int cols, double mat[N][N]) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i][j] = rand() % 10;
}

void multiplyPartialMatrices(int startRow, int numRows, int size) {
    for (int i = 0; i < numRows; ++i)
        for (int j = 0; j < size; ++j) {
            matrixC[startRow + i][j] = 0.0;
            for (int k = 0; k < size; ++k)
                matrixC[startRow + i][j] += matrixA[startRow + i][k] * matrixB[k][j];
        }
}

int main(int argc, char** argv) {
    int rank, numProcs, index, elementsPerProc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    std::vector<int> sizes = {10, 100, 1000, 2000};

    for (int size : sizes) {
        if (rank == 0) {
            generateMatrix(size, size, matrixA);
            generateMatrix(size, size, matrixB);

            for (int i = 0; i < size; ++i)
                for (int j = 0; j < size; ++j)
                    matrixC[i][j] = 0.0;

            auto startTime = std::chrono::high_resolution_clock::now();

            elementsPerProc = size / numProcs;
            index = elementsPerProc;

            for (int i = 1; i < numProcs - 1; ++i) {
                MPI_Send(&index, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&elementsPerProc, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&matrixA[index][0], elementsPerProc * size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&matrixB, size * size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                index += elementsPerProc;
            }

            int remaining = size - index;
            MPI_Send(&index, 1, MPI_INT, numProcs - 1, 0, MPI_COMM_WORLD);
            MPI_Send(&remaining, 1, MPI_INT, numProcs - 1, 0, MPI_COMM_WORLD);
            MPI_Send(&matrixA[index][0], remaining * size, MPI_DOUBLE, numProcs - 1, 0, MPI_COMM_WORLD);
            MPI_Send(&matrixB, size * size, MPI_DOUBLE, numProcs - 1, 0, MPI_COMM_WORLD);

            multiplyPartialMatrices(0, elementsPerProc, size);

            for (int i = 1; i < numProcs; ++i) {
                int recvIndex, rowsReceived;
                MPI_Recv(&recvIndex, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&rowsReceived, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&matrixC[recvIndex][0], rowsReceived * size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = endTime - startTime;

            std::cout << "Matrix size: " << size << "x" << size
                      << ", Execution time: " << duration.count() << " s" << std::endl;
        } else {
            int startRow, numRows;
            MPI_Recv(&startRow, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&numRows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrixA[startRow][0], numRows * size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrixB, size * size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

            multiplyPartialMatrices(startRow, numRows, size);

            MPI_Send(&startRow, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&numRows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&matrixC[startRow][0], numRows * size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
