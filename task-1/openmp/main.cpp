#include <stdio.h>
#include <omp.h>

int main() {
    int num_threads = 4;
    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("Hello from thread %d\n", thread_id);
        #pragma omp barrier
        if (thread_id == 0) {
            printf("All threads have reached the barrier.\n");
        }
    }
    return 0;
}