#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> generateMatrix(int rows, int cols) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = rand() % 9 + 1;
    return matrix;
}

std::vector<std::vector<int>> multiplyMatrices(
    const std::vector<std::vector<int>>& matrixA,
    const std::vector<std::vector<int>>& matrixB) {

    int rowsA = matrixA.size();
    int colsA = matrixA[0].size();
    int rowsB = matrixB.size();
    int colsB = matrixB[0].size();

    if (colsA != rowsB) {
        std::cerr << "Matrix multiplication error: incompatible dimensions."
                  << std::endl;
        std::exit(1);
    }

    std::vector<std::vector<int>> result(rowsA, std::vector<int>(colsB, 0));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }

    return result;
}

int main() {
    std::vector<std::pair<int, int>> matrixSizes = {
        {10, 10}, {100, 100}, {1000, 1000}, {2000, 2000}};

    for (const auto& size : matrixSizes) {
        int rowsA = size.first;
        int colsA = size.second;
        int rowsB = colsA;
        int colsB = size.first;

        std::vector<std::vector<int>> matrixA = generateMatrix(rowsA, colsA);
        std::vector<std::vector<int>> matrixB = generateMatrix(rowsB, colsB);

        double startTime = omp_get_wtime();
        std::vector<std::vector<int>> result = multiplyMatrices(matrixA, matrixB);
        double endTime = omp_get_wtime();

        std::cout << "Matrix sizes: " << rowsA << "x" << colsA << " * "
                  << rowsB << "x" << colsB
                  << ", Execution time: " << (endTime - startTime)
                  << " s" << std::endl;
    }

    return 0;
}
