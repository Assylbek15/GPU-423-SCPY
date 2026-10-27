"""Practice 10: matrix multiplication followed by reduction.

Fill two NxN integer matrices A and B in [-5,5] on independent streams.
Compute C=A@B and S=sum of every element of C using shared-memory reduction.
Implement CPU multiplication, verify C and S, and time the complete calculation.
Show N=4 results, then use N=1<<6. Use 256 threads, warp_size*SM_count blocks,
grid-stride loops and pinned asynchronous copies. Release device memory.
Run on a CUDA GPU: python midterm_variant_10_matrix_product_sum.py
GPU time includes launch/synchronization; CPU time includes Python overhead.
"""

from time import perf_counter
import numpy as np
from numba import cuda, int64


@cuda.jit
def fill_matrix(a, multiplier, offset):
    n = a.shape[1]
    for i in range(cuda.grid(1), a.size, cuda.gridsize(1)):
        a[i // n, i % n] = (multiplier * i + offset) % 11 - 5


@cuda.jit
def multiply_kernel(a, b, c):
    n = a.shape[0]
    for i in range(cuda.grid(1), c.size, cuda.gridsize(1)):
        row, col = i // n, i % n
        total = int64(0)
        for k in range(n):
            total += int64(a[row, k]) * b[k, col]
        c[row, col] = total


def multiply_cpu(a, b, c):
    n = a.shape[0]
    for row in range(n):
        for col in range(n):
            total = 0
            for k in range(n):
                total += int(a[row, k]) * int(b[k, col])
            c[row, col] = total


@cuda.jit
def reduce_sum(values, partials):
    shared = cuda.shared.array(0, dtype=int64)
    tid = cuda.threadIdx.x
    subtotal = int64(0)
    for i in range(cuda.grid(1), values.size, cuda.gridsize(1)):
        subtotal += values[i]
    shared[tid] = subtotal
    cuda.syncthreads()
    step = 1
    while step < cuda.blockDim.x:
        if tid % (2 * step) == 0:
            shared[tid] += shared[tid + step]
        cuda.syncthreads()
        step *= 2
    if tid == 0:
        partials[cuda.blockIdx.x] = shared[0]


def calculate_gpu(a, b, c, partials, total, blocks, threads, stream):
    shared_bytes = threads * np.dtype(np.int64).itemsize
    multiply_kernel[blocks, threads, stream](a, b, c)
    # reshape is a device view: it does not copy the matrix to the CPU.
    reduce_sum[blocks, threads, stream, shared_bytes](c.reshape(c.size), partials)
    reduce_sum[1, threads, stream, shared_bytes](partials, total)


def verify(cpu_c, gpu_c, cpu_sum, gpu_sum):
    correct = np.array_equal(cpu_c, gpu_c) and cpu_sum == gpu_sum
    print("Verification:", "PASS" if correct else "FAIL")
    return correct


def run_case(n, blocks, threads):
    stream_a, stream_b = cuda.stream(), cuda.stream()
    a = cuda.pinned_array((n, n), np.int32)
    b = cuda.pinned_array((n, n), np.int32)
    cpu_c = np.empty((n, n), np.int64)
    gpu_c = cuda.pinned_array((n, n), np.int64)
    gpu_sum = cuda.pinned_array(1, np.int64)
    d_a = cuda.device_array((n, n), np.int32, stream=stream_a)
    d_b = cuda.device_array((n, n), np.int32, stream=stream_b)
    d_c = cuda.device_array((n, n), np.int64, stream=stream_a)
    partials = cuda.device_array(blocks, np.int64, stream=stream_a)
    total = cuda.device_array(1, np.int64, stream=stream_a)

    fill_matrix[blocks, threads, stream_a](d_a, 3, 1)
    fill_matrix[blocks, threads, stream_b](d_b, 7, 4)
    d_a.copy_to_host(a, stream=stream_a)
    d_b.copy_to_host(b, stream=stream_b)
    stream_a.synchronize()
    stream_b.synchronize()

    start = perf_counter()
    multiply_cpu(a, b, cpu_c)
    cpu_sum = int(np.sum(cpu_c, dtype=np.int64))
    cpu_ms = (perf_counter() - start) * 1000

    calculate_gpu(d_a, d_b, d_c, partials, total, blocks, threads, stream_a)
    stream_a.synchronize()  # Finish warm-up before timing.
    start = perf_counter()
    calculate_gpu(d_a, d_b, d_c, partials, total, blocks, threads, stream_a)
    stream_a.synchronize()
    gpu_ms = (perf_counter() - start) * 1000

    d_c.copy_to_host(gpu_c, stream=stream_a)
    total.copy_to_host(gpu_sum, stream=stream_a)
    stream_a.synchronize()
    if n == 4:
        print("A =\n", a)
        print("B =\n", b)
        print("CPU C =\n", cpu_c)
        print("GPU C =\n", gpu_c)
    print(f"N={n}: CPU sum={cpu_sum}; GPU sum={gpu_sum[0]}")
    print(f"CPU calculation: {cpu_ms:.3f} ms; GPU calculation: {gpu_ms:.3f} ms")
    return verify(cpu_c, gpu_c, cpu_sum, int(gpu_sum[0]))


def main():
    device = cuda.get_current_device()
    threads = 256
    blocks = device.WARP_SIZE * device.MULTIPROCESSOR_COUNT
    print("Hardware name:", device.name)
    print(f"Blocks: {blocks}; threads per block: {threads}")
    try:
        run_case(4, blocks, threads)
        run_case(1 << 6, blocks, threads)
    finally:
        cuda.close()


if __name__ == "__main__":
    main()
