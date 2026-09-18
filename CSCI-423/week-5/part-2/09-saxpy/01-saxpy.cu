#include <stdio.h>
#include <stdlib.h>

#define N (2048 * 2048) // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 200,000 ns.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

void checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        exit(1);
    }
}

__global__ void initVectors(int *a, int *b)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride)
    {
        a[i] = 2;
        b[i] = 1;
    }
}

__global__ void saxpy(const int4 *__restrict__ a,
                      const int4 *__restrict__ b,
                      int4 *__restrict__ c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vector_count = N / 4;

    for (int i = tid; i < vector_count; i += stride)
    {
        int4 a_value = a[i];
        int4 b_value = b[i];

        c[i] = make_int4(2 * a_value.x + b_value.x,
                         2 * a_value.y + b_value.y,
                         2 * a_value.z + b_value.z,
                         2 * a_value.w + b_value.w);
    }
}

int main()
{
    int *a, *b, *c;

    size_t size = N * sizeof(int); // The total number of bytes per vector

    checkCuda(cudaMallocManaged(&a, size));
    checkCuda(cudaMallocManaged(&b, size));
    checkCuda(cudaMallocManaged(&c, size));

    int device_id;
    int number_of_sms;
    checkCuda(cudaGetDevice(&device_id));
    checkCuda(cudaDeviceGetAttribute(&number_of_sms,
                                     cudaDevAttrMultiProcessorCount,
                                     device_id));

    checkCuda(cudaMemPrefetchAsync(a, size, device_id));
    checkCuda(cudaMemPrefetchAsync(b, size, device_id));
    checkCuda(cudaMemPrefetchAsync(c, size, device_id));

    int threads_per_block = 256;
    int number_of_blocks = 32 * number_of_sms;

    initVectors<<<number_of_blocks, threads_per_block>>>(a, b);
    checkCuda(cudaGetLastError());

    saxpy<<<number_of_blocks, threads_per_block>>>(
        reinterpret_cast<const int4 *>(a),
        reinterpret_cast<const int4 *>(b),
        reinterpret_cast<int4 *>(c));
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemPrefetchAsync(c, size, cudaCpuDeviceId));
    checkCuda(cudaDeviceSynchronize());

    // Print out the first and last 5 values of c for a quality check
    for (int i = 0; i < 5; ++i)
        printf("c[%d] = %d, ", i, c[i]);
    printf("\n");
    for (int i = N - 5; i < N; ++i)
        printf("c[%d] = %d, ", i, c[i]);
    printf("\n");

    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
    checkCuda(cudaFree(c));

    return 0;
}
