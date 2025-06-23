#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>

void fillRandom(int* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = rand() % 10;
    }
}

int computeSum(const int* data, int size) {
    int total = 0;
    for (int i = 0; i < size; ++i) {
        total += data[i];
    }
    return total;
}

int main(int argc, char* argv[]) {
    unsigned int seed = 42;
    srand(seed);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> testSizes = {10, 1000, 10000000};

    for (int currSize : testSizes) {
        int baseChunk = currSize / size;
        int remainder = currSize % size;

        int* localData = nullptr;
        int localSize = (rank < remainder) ? baseChunk + 1 : baseChunk;

        if (rank == 0) {
            std::vector<int> full_data(currSize);
            fillRandom(full_data.data(), currSize);

            auto start_time = std::chrono::high_resolution_clock::now();

            int offset = 0;
            for (int i = 1; i < size; ++i) {
                int chunkSize = (i < remainder) ? baseChunk + 1 : baseChunk;
                MPI_Send(&chunkSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(full_data.data() + offset, chunkSize, MPI_INT, i, 0, MPI_COMM_WORLD);
                offset += chunkSize;
            }

            int sum = computeSum(full_data.data(), localSize);
            int received_sum = 0;

            for (int i = 1; i < size; ++i) {
                MPI_Recv(&received_sum, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sum += received_sum;
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;

            std::cout << "Array size: " << currSize
                      << ", Total sum: " << sum
                      << ", Time: " << duration.count() << " s." << std::endl;
        } else {
            MPI_Recv(&localSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            localData = new int[localSize];
            MPI_Recv(localData, localSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int partial_sum = computeSum(localData, localSize);
            MPI_Send(&partial_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            delete[] localData;
        }
    }

    MPI_Finalize();
    return 0;
}
