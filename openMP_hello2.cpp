#include <iostream>
#include <omp.h>
using namespace std;

int main() {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int n   = omp_get_num_threads();
        #pragma omp critical
        cout << "hi !! from thread " << tid << " / " << n << endl;
    }
}
