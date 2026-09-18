#include <stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

__global__ void loop(int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N)
  {
    printf("This is iteration number %d\n", i);
  }
}

int main()
{
  /*
   * When refactoring `loop` to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * For this exercise, be sure to use more than 1 block in
   * the execution configuration.
   */

  int N = 10;
  int threads_per_block = 5;
  int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  loop<<<number_of_blocks, threads_per_block>>>(N);
  cudaDeviceSynchronize();
}
