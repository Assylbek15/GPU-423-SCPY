"""Practice 07: multiply NxN matrix A by N-element x, y[i] = sum_j A[i,j]*x[j].

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
def fill_vector(x, multiplier, offset):
    for i in range(cuda.grid(1), x.size, cuda.gridsize(1)):
        x[i] = (multiplier * i + offset) % 11 - 5


@cuda.jit
def matrix_vector_kernel(a, x, y):
    for row in range(cuda.grid(1), y.size, cuda.gridsize(1)):
        total = int64(0)
        for col in range(x.size):
            total += int64(a[row, col]) * int64(x[col])
        y[row] = total


def matrix_vector_cpu(a, x, y):
    for row in range(y.size):
        total = 0
        for col in range(x.size):
            total += int(a[row, col]) * int(x[col])
        y[row] = total


def verify_result(cpu_y, gpu_y):
    passed = bool(np.array_equal(cpu_y, gpu_y))
    print("Verification:", "PASS" if passed else "FAIL")
    return passed


def run_case(n, blocks, threads):
    stream_a, stream_x = cuda.stream(), cuda.stream()
    a = cuda.pinned_array((n, n), dtype=np.int32)
    x = cuda.pinned_array(n, dtype=np.int32)
    gpu_y = cuda.pinned_array(n, dtype=np.int64)
    cpu_y = np.empty(n, dtype=np.int64)
    d_a = cuda.device_array((n, n), dtype=np.int32, stream=stream_a)
    d_x = cuda.device_array(n, dtype=np.int32, stream=stream_x)
    d_y = cuda.device_array(n, dtype=np.int64, stream=stream_a)

    fill_matrix[blocks, threads, stream_a](d_a, 3, 1)
    fill_vector[blocks, threads, stream_x](d_x, 7, 4)
    d_a.copy_to_host(a, stream=stream_a)
    d_x.copy_to_host(x, stream=stream_x)
    # Wait for both independent fills and their asynchronous host copies.
    stream_a.synchronize()
    stream_x.synchronize()

    start = perf_counter()
    matrix_vector_cpu(a, x, cpu_y)
    cpu_ms = (perf_counter() - start) * 1000

    matrix_vector_kernel[blocks, threads, stream_a](d_a, d_x, d_y)
    stream_a.synchronize()  # Compile and warm up before timing.
    start = perf_counter()
    matrix_vector_kernel[blocks, threads, stream_a](d_a, d_x, d_y)
    stream_a.synchronize()
    gpu_ms = (perf_counter() - start) * 1000
    d_y.copy_to_host(gpu_y, stream=stream_a)
    stream_a.synchronize()

    print(f"\nN = {n}")
    if n == 4:
        print("A =\n", a)
        print("x =", x)
        print("CPU y =", cpu_y)
        print("GPU y =", gpu_y)
    passed = verify_result(cpu_y, gpu_y)
    print(f"CPU matrix-vector multiplication (Python): {cpu_ms:.6f} ms")
    print(f"GPU matrix-vector multiplication:          {gpu_ms:.6f} ms")
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
