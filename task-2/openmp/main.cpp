#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>

std::vector<int> createRandomVector(int length) {
    std::vector<int> data(length);
    for (int i = 0; i < length; ++i) {
        data[i] = rand() % 10;
    }
    return data;
}

int main() {
    const unsigned int randomSeed = 42;
    srand(randomSeed);

    std::vector<int> arraySizes = {10, 1000, 10000000};

    for (int currentSize : arraySizes) {
        std::vector<int> inputData = createRandomVector(currentSize);

        double timeStart = omp_get_wtime();

        int totalSum = 0;
#pragma omp parallel for reduction(+ : totalSum)
        for (int i = 0; i < currentSize; ++i) {
            totalSum += inputData[i];
        }

        double timeEnd = omp_get_wtime();

        std::cout << "Array size: " << currentSize
                  << ", Computed sum: " << totalSum
                  << ", Execution time: " << (timeEnd - timeStart)
                  << " s" << std::endl;
    }

    return 0;
}
