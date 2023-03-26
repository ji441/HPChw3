#include <algorithm>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
// g++ -std=c++11 -O3 -fopenmp -march=native gs2D-omp.cpp  && ./a.out
// g++ -std=c++11 -O0 -fopenmp -march=native gs2D-omp.cpp -g
// valgrind --leak-check=full ./a.out
void parallel(int &nthreads, long &N)
{
    double h = 1.0 / (N + 1);
    double *u = new double[(N + 2) * (N + 2)];
    // initialize
    for (long i = 0; i < (N + 2) * (N + 2); i++)
    {
        u[i] = 0;
    }
    Timer tt;
    tt.tic();
#if defined(_OPENMP)
    omp_set_num_threads(nthreads);
    for (int i = 0; i < 1000; i++) // 1000 iterations
    {

#pragma omp parallel for // update red first
        for (int r = 1; r < N + 1; r++)
        {
            int c;
            if (r % 2 == 1)
            {
                c = 1;
            }
            else
            {
                c = 2;
            }
            for (c; c < N + 1; c += 2)
            {
                u[r + c * (N + 2)] = 1 / 4 * (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]);
            }
        }
#pragma omp parallel for // update black then
        for (int r = 1; r < N + 1; r++)
        {
            int c;
            if (r % 2 == 1)
            {
                c = 2;
            }
            else
            {
                c = 1;
            }
            for (c; c < N + 1; c += 2)
            {
                u[r + c * (N + 2)] = 1 / 4 * (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]);
            }
        }
    }

    printf("with %d threads,N = %ld,the time for 1000 iterations are: %.6f\n", nthreads, N, tt.toc());
#else
    for (int i = 0; i < 1000; i++)
    {

        // update red first
        for (int r = 1; r < N + 1; r++)
        {
            int c;
            if (r % 2 == 1)
            {
                c = 1;
            }
            else
            {
                c = 2;
            }
            for (c; c < N + 1; c += 2)
            {
                u[r + c * (N + 2)] = 1 / 4 * (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]);
            }
        }
        // update black then
        for (int r = 1; r < N + 1; r++)
        {
            int c;
            if (r % 2 == 1)
            {
                c = 2;
            }
            else
            {
                c = 1;
            }
            for (c; c < N + 1; c += 2)
            {
                u[r + c * (N + 2)] = 1 / 4 * (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]);
            }
        }
    }
    printf("with one threads,N = %ld,the time for 1000 iterations are: %.6f\n", N, tt.toc());
#endif
    delete[] u;
}
int main()
{
    int tarr[] = {1, 8, 16, 32};
    long narr[] = {100, 1000, 2000};
    for (int numthreads : tarr)
    {
        for (long N : narr)
        {
            parallel(numthreads, N);
        }
    }
}