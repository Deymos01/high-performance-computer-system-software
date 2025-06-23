#include <omp.h>
#include <cmath>
#include <iostream>
#include <vector>

double computeFunction(double x, double y) {
    return x * (sin(x) + cos(y));
}

void computePartialDerivativeX(const std::vector<std::vector<double>>& input,
                               std::vector<std::vector<double>>& output,
                               double dx) {
    int numRows = input.size();
    int numCols = input[0].size();

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            if (j == 0) {
                output[i][j] = (input[i][j + 1] - input[i][j]) / dx;
            } else if (j == numCols - 1) {
                output[i][j] = (input[i][j] - input[i][j - 1]) / dx;
            } else {
                output[i][j] = (input[i][j + 1] - input[i][j - 1]) / (2 * dx);
            }
        }
    }
}

int main() {
    std::vector<int> gridSizes = {10, 100, 1000, 10000};
    double dx = 0.01;

    for (int size : gridSizes) {
        int rows = size;
        int cols = size;

        std::vector<std::vector<double>> grid(rows, std::vector<double>(cols));
        std::vector<std::vector<double>> derivative(rows, std::vector<double>(cols));

#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                grid[i][j] = computeFunction(i * dx, j * dx);
            }
        }

        double startTime = omp_get_wtime();

        computePartialDerivativeX(grid, derivative, dx);

        double endTime = omp_get_wtime();

        std::cout << "Grid size: " << rows << "x" << cols
                  << ", Execution time: " << (endTime - startTime) << " seconds"
                  << std::endl;
    }

    return 0;
}
