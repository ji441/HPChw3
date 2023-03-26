// g++ -std=c++11 -O3 -fopenmp -march=native omp-scan.cpp && ./a.out
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
using namespace std;
// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long *prefix_sum, const long *A, long n)
{
  if (n == 0)
    return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++)
  {
    prefix_sum[i] = prefix_sum[i - 1] + A[i - 1];
  }
}

void scan_omp(long *prefix_sum, const long *A, long n)
{

  int p = 64;
  printf("%d threads used:\n", p);
  int t = omp_get_thread_num();
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  int chunksize = ceil(n / double(p));
  // scan each subchunk,write the result into shared vector prefix_sum

#pragma omp parallel for num_threads(p)
  for (int i = 0; i < p; i++)
  {
    for (int j = i * chunksize + 1; j < chunksize * (i + 1); j++)
    {
      if (j < n)
      {
        prefix_sum[j] = prefix_sum[j - 1] + A[j - 1];
      }
    }
  }

  for (int i = 1; i < p; i++)
  {
    long correction = prefix_sum[i * chunksize - 1] + A[i * chunksize - 1];
    int stater = i * chunksize;

#pragma omp parallel for num_threads(p)
    for (int j = stater; j < chunksize * (i + 1); j++)
    {
      if (j < n)
      {
        prefix_sum[j] += correction;
      }
    }
  }
}

int main()
{
  long N = 100000000;
  long *A = (long *)malloc(N * sizeof(long));
  long *B0 = (long *)malloc(N * sizeof(long));
  long *B1 = (long *)malloc(N * sizeof(long));
  for (long i = 0; i < N; i++)
    A[i] = rand();
  for (long i = 0; i < N; i++)
    B1[i] = 0;

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++)
    err = max(err, abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
