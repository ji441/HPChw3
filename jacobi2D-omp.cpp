#include <algorithm>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
// g++ -std=c++11 -O3 -fopenmp -march=native jacobi2D-omp.cpp && ./a.out
// g++ -std=c++11 -O0 -fopenmp -march=native jacobi2D-omp.cpp -g
// valgrind --leak-check=full ./a.out
double fnorm(int nthreads, double h, long N, double *u)
{
    double sum = 0;
#if defined(_OPENMP)
    omp_set_num_threads(nthreads);
#pragma omp parallel for reduction(+ \
                                   : sum)
    for (int r = 1; r < N + 1; r++)
    {
        for (int c = 1; c < N + 1; c++)
        {
            sum += pow(abs(1 + (u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] - 4 * u[r + c * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]) / (h * h)), 2.0);
        }
    }
#else
    for (int r = 1; r < N + 1; r++)
    {
        for (int c = 1; c < N + 1; c++)
        {
            sum += pow(abs(1 + (u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] - 4 * u[r + c * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]) / (h * h)), 2.0);
        }
    }
#endif

    return sqrt(sum);
}
void parallel(int numthreads, long N)
{
    double *u = new double[(N + 2) * (N + 2)];
    double h = 1.0 / (N + 1);
    // initialize
    for (long i = 0; i < (N + 2) * (N + 2); i++)
    {
        u[i] = 0;
    }
    double error;
    // timer begins
    Timer tt;
    tt.tic();
// fixed iteration,omp version
#if defined(_OPENMP)

    omp_set_num_threads(numthreads); // set threads used
    for (int i = 0; i < 1000; i++)
    {
        double *u_new = new double[(N + 2) * (N + 2)]; // container for new u values
#pragma omp parallel for
        for (int r = 1; r < N + 1; r++)
        {
            for (int c = 1; c < N + 1; c++)
            {
                u_new[r + c * (N + 2)] = 1.0 / 4 * (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]);
            }
        }
#pragma omp parallel for
        for (int r = 1; r < N + 1; r++) // write the new vlaue into u
        {
            for (int c = 1; c < N + 1; c++)
            {
                u[r + c * (N + 2)] = u_new[r + c * (N + 2)];
            }
        }
        delete[] u_new;
        // check for convergence

        error = fnorm(numthreads, h, N, u);
        if (error < N / 10000.0)
        {
            break;
        }
    }
    printf("with %d threads,N = %ld,the time for 1000 iterations are: %.6f, error is: %f\n", numthreads, N, tt.toc(), error);
#else
    for (int i = 0; i < 1000; i++)
    {
        double *u_new = new double[(N + 2) * (N + 2)];
        for (int r = 1; r < N + 1; r++)
        {
            for (int c = 1; c < N + 1; c++)
            {
                u_new[r + c * (N + 2)] = 1.0 / 4 * (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]);
            }
        }
        for (int r = 1; r < N + 1; r++)
        {
            for (int c = 1; c < N + 1; c++)
            {
                u[r + c * (N + 2)] = u_new[r + c * (N + 2)];
            }
        }
        delete[] u_new;
        // check for convergence

        error = fnorm(numthreads, h, N, u);
        if (error < N / 10000.0)
        {
            break;
        }
    }
    printf("with one thread,N = %ld,the time for 1000 iterations are: %.6f\n,error is: %f\n", N, tt.toc(), error);
#endif
    delete[] u;
}
int main()
{
    int tarr[] = {1, 8, 16, 32};
    long narr[] = {10, 100, 1000};
    for (int numthreads : tarr)
    {
        for (long N : narr)
        {
            parallel(numthreads, N);
        }
    }
}