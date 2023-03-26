#include <algorithm>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
// g++ -std=c++11 -O2 -fopenmp -march=native jacobi2D-omp.cpp && ./a.out
// g++ -std=c++11 -O0 -fopenmp -march=native jacobi2D-omp.cpp -g
// valgrind --leak-check=full ./a.out
void parallel(int numthreads, long N)
{
    double *u = new double[(N + 2) * (N + 2)];
    double h = 1.0 / (N + 1);
    // initialize
    for (long i = 0; i < (N + 2) * (N + 2); i++)
    {
        u[i] = 0;
    }
    // timer begins
    Timer tt;
    tt.tic();
// fixed iteration,omp version
#if defined(_OPENMP)

    omp_set_num_threads(numthreads); // set threads used
    for (int i = 0; i < 1000; i++)
    {
        double *u_new = new double[(N + 2) * (N + 2)];
#pragma omp parallel for
        for (int r = 1; r < N + 1; r++)
        {
            for (int c = 1; c < N + 1; c++)
            {
                u_new[r + c * (N + 2)] = 1 / 4 * (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]);
            }
        }
#pragma omp parallel for
        for (int r = 1; r < N + 1; r++)
        {
            for (int c = 1; c < N + 1; c++)
            {
                u[r + c * (N + 2)] = u_new[r + c * (N + 2)];
            }
        }
        delete[] u_new;
    }
    printf("with %d threads,N = %ld,the time for 1000 iterations are: %.6f\n", numthreads, N, tt.toc());
#else
    for (int i = 0; i < 1000; i++)
    {
        double u_new[(N + 2) * (N + 2)];
        for (int r = 1; r < N + 1; r++)
        {
            for (int c = 1; c < N + 1; c++)
            {
                u_new[r + c * (N + 2)] = 1 / 4 * (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]);
            }
        }
        for (int r = 1; r < N + 1; r++)
        {
            for (int c = 1; c < N + 1; c++)
            {
                u[r + c * (N + 2)] = u_new[r + c * (N + 2)];
            }
        }
    }
    printf("with one thread,N = %ld,the time for 1000 iterations are: %.6f\n", N, tt.toc());
#endif
    delete[] u;
}
int main()
{
    int tarr[] = {1, 8, 16, 32, 64};
    long narr[] = {100, 1000, 2000};
    for (int numthreads : tarr)
    {
        for (long N : narr)
        {
            parallel(numthreads, N);
        }
    }
}