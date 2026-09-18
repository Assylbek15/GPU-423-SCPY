#include <stdio.h>
#include <stdlib.h>

void checkCuda(cudaError_t result, const char *operation)
{
  if (result != cudaSuccess)
  {
    fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
    exit(1);
  }
}

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__ void addVectorsInto(float *result, const float *a, const float *b, int N)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  checkCuda(cudaMallocManaged(&a, size), "allocate a");
  checkCuda(cudaMallocManaged(&b, size), "allocate b");
  checkCuda(cudaMallocManaged(&c, size), "allocate c");

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  const int threads_per_block = 256;
  const int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  addVectorsInto<<<number_of_blocks, threads_per_block>>>(c, a, b, N);
  checkCuda(cudaGetLastError(), "addVectorsInto launch");
  checkCuda(cudaDeviceSynchronize(), "addVectorsInto execution");

  checkElementsAre(7, c, N);

  checkCuda(cudaFree(a), "free a");
  checkCuda(cudaFree(b), "free b");
  checkCuda(cudaFree(c), "free c");
}
