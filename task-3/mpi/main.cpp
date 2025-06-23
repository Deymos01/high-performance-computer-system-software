#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

constexpr int maxSize = 10000;

double matrixA[maxSize][maxSize];
double matrixB[maxSize][maxSize];

double computeFunction(double x, double y) {
    return x * (sin(x) + cos(y));
}

constexpr double dx = 0.01;

void generateMatrix(int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrixA[i][j] = computeFunction(i * dx, j * dx);
}

void computeDerivativeX(int startRow, int numRows, int cols) {
    for (int i = startRow; i < startRow + numRows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (j == 0) {
                matrixB[i][j] = (matrixA[i][j + 1] - matrixA[i][j]) / dx;
            } else if (j == cols - 1) {
                matrixB[i][j] = (matrixA[i][j] - matrixA[i][j - 1]) / dx;
            } else {
                matrixB[i][j] = (matrixA[i][j + 1] - matrixA[i][j - 1]) / (2 * dx);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, numProcesses;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    std::vector<int> gridSizes = {10, 100, 1000, 10000};

    for (auto size : gridSizes) {
        int rows = size;
        int cols = size;

        int rowsPerProcess = rows / numProcesses;
        int remainingRows = rows % numProcesses;

        if (rank == 0) {
            generateMatrix(rows, cols);

            auto startTime = std::chrono::high_resolution_clock::now();

            for (int proc = 1; proc < numProcesses; ++proc) {
                int startRow = proc * rowsPerProcess;
                int numRowsToSend = (proc == numProcesses - 1) ? rowsPerProcess + remainingRows : rowsPerProcess;

                MPI_Send(&numRowsToSend, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
                MPI_Send(&startRow, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
                MPI_Send(&matrixA[startRow][0], numRowsToSend * cols, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
            }

            computeDerivativeX(0, rowsPerProcess, cols);

            for (int proc = 1; proc < numProcesses; ++proc) {
                int numRowsReceived, startRowReceived;
                MPI_Recv(&numRowsReceived, 1, MPI_INT, proc, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&startRowReceived, 1, MPI_INT, proc, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&matrixB[startRowReceived][0], numRowsReceived * cols, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, &status);
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = endTime - startTime;

            std::cout << "Grid size: " << rows << "x" << cols
                      << ", Execution time: " << elapsed.count() << " seconds" << std::endl;
        } else {
            int numRowsToProcess, startRow;
            MPI_Recv(&numRowsToProcess, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&startRow, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            MPI_Recv(&matrixA[startRow][0], numRowsToProcess * cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

            computeDerivativeX(startRow, numRowsToProcess, cols);

            MPI_Send(&numRowsToProcess, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&startRow, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&matrixB[startRow][0], numRowsToProcess * cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
