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
// g++ -std=c++11 -O3 -march=native gs2D-omp.cpp  && ./a.out
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
    double error; // original error is N
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
                u[r + c * (N + 2)] = (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]) / 4.0;
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
                u[r + c * (N + 2)] = (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]) / 4.0;
            }
        }
        // check for convergence

        error = fnorm(nthreads, h, N, u);
        if (error < N / 10000.0)
        {
            break;
        }
    }

    printf("with %d threads,N = %ld,the time for 1000 iterations are: %.6f, error is: %f\n", nthreads, N, tt.toc(), error);

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
                u[r + c * (N + 2)] = (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]) / 4.0;
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
                u[r + c * (N + 2)] = 1.0 / 4 * (h * h + u[r - 1 + c * (N + 2)] + u[r + (c - 1) * (N + 2)] + u[r + 1 + c * (N + 2)] + u[r + (c + 1) * (N + 2)]);
            }
        }
        // check for convergence
        error = fnorm(nthreads, h, N, u);
        if (error < N / 10000.0)
        {
            break;
        }
    }

    printf("with one threads,N = %ld,the time for 1000 iterations are: %.6f, error is: %f\n", N, tt.toc(), error);
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