#include <stdio.h>
#include "mpi.h"

int main(int argc, char **argv)
{	
	const int MAX = 3;
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	for(int i = 0; i < MAX; i++)
	{
		printf("Message from process: %d, size: %d\n", rank, size);
	}

	MPI_Finalize();
	
	return 0;
}