"""Practice 06: multiply two NxN integer matrices, C[i,j] = sum_k A[i,k]*B[k,j].

Use N=4 to display inputs and CPU/GPU output; repeat with N=1<<6 for timings.
Initialize values in [-5,5] on independent streams and copy to pinned host
memory. Use grid-stride kernels, 256 threads, and WARP_SIZE*SM_count blocks.
Verify every int64 result against plain Python loops and release CUDA resources.
GPU time includes launch/synchronization, excluding compilation and transfers;
CPU time includes Python overhead. Requires NumPy, Numba, and a CUDA GPU.
"""

from time import perf_counter
import numpy as np
from numba import cuda, int64


@cuda.jit
def fill_matrix(a, multiplier, offset):
    n = a.shape[1]
    for index in range(cuda.grid(1), a.size, cuda.gridsize(1)):
        row, col = index // n, index % n
        a[row, col] = (multiplier * index + offset) % 11 - 5


@cuda.jit
def matmul_kernel(a, b, c):
    n = a.shape[0]
    # Flatten output indices so the required one-dimensional grid stays fixed.
    for index in range(cuda.grid(1), c.size, cuda.gridsize(1)):
        row, col = index // n, index % n
        total = int64(0)
        for k in range(n):
            total += int64(a[row, k]) * int64(b[k, col])
        c[row, col] = total


def matmul_cpu(a, b, c):
    n = a.shape[0]
    for row in range(n):
        for col in range(n):
            total = 0
            for k in range(n):
                total += int(a[row, k]) * int(b[k, col])
            c[row, col] = total


def verify_result(cpu_c, gpu_c):
    passed = bool(np.array_equal(cpu_c, gpu_c))
    print("Verification:", "PASS" if passed else "FAIL")
    return passed


def run_case(n, blocks, threads):
    stream_a, stream_b = cuda.stream(), cuda.stream()
    a = cuda.pinned_array((n, n), dtype=np.int32)
    b = cuda.pinned_array((n, n), dtype=np.int32)
    gpu_c = cuda.pinned_array((n, n), dtype=np.int64)
    cpu_c = np.empty((n, n), dtype=np.int64)
    d_a = cuda.device_array((n, n), dtype=np.int32, stream=stream_a)
    d_b = cuda.device_array((n, n), dtype=np.int32, stream=stream_b)
    d_c = cuda.device_array((n, n), dtype=np.int64, stream=stream_a)

    fill_matrix[blocks, threads, stream_a](d_a, 3, 1)
    fill_matrix[blocks, threads, stream_b](d_b, 7, 4)
    d_a.copy_to_host(a, stream=stream_a)
    d_b.copy_to_host(b, stream=stream_b)
    # Both fills and pinned copies must finish before either input is consumed.
    stream_a.synchronize()
    stream_b.synchronize()

    start = perf_counter()
    matmul_cpu(a, b, cpu_c)
    cpu_ms = (perf_counter() - start) * 1000

    matmul_kernel[blocks, threads, stream_a](d_a, d_b, d_c)
    stream_a.synchronize()  # Compile and warm up before timing.
    start = perf_counter()
    matmul_kernel[blocks, threads, stream_a](d_a, d_b, d_c)
    stream_a.synchronize()
    gpu_ms = (perf_counter() - start) * 1000
    d_c.copy_to_host(gpu_c, stream=stream_a)
    stream_a.synchronize()

    print(f"\nN = {n}")
    if n == 4:
        print("A =\n", a)
        print("B =\n", b)
        print("CPU C =\n", cpu_c)
        print("GPU C =\n", gpu_c)
    passed = verify_result(cpu_c, gpu_c)
    print(f"CPU matrix multiplication (Python): {cpu_ms:.6f} ms")
    print(f"GPU matrix multiplication:          {gpu_ms:.6f} ms")
    # Local buffer and stream references are released on return.
    return passed


def main():
    try:
        device = cuda.get_current_device()
        name = device.name.decode() if isinstance(device.name, bytes) else device.name
        threads = 256
        blocks = device.WARP_SIZE * device.MULTIPROCESSOR_COUNT
        print(f"GPU: {name}")
        print(f"Blocks: {blocks}; threads per block: {threads}")
        run_case(4, blocks, threads)
        run_case(1 << 6, blocks, threads)
    finally:
        cuda.close()  # Release the context and deferred device allocations.


if __name__ == "__main__":
    main()
