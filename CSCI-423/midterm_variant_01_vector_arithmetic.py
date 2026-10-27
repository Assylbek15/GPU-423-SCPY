"""Practice 01: vector arithmetic and reduction.

Fill integer vectors A and B in [-5,5] on independent streams. Implement
separate device functions and kernels for A+B, A-B, and A*B. Sum each result
with shared-memory reduction and verify every element and sum on the CPU.
Show N=10 results; time CPU/GPU multiplication for N=2<<20.
Use grid-stride loops, 256 threads, warp_size*SM_count blocks, pinned copies.
Run on a CUDA GPU: python midterm_variant_01_vector_arithmetic.py
GPU time includes launch/synchronization; CPU time includes Python overhead.
"""

from time import perf_counter
import numpy as np
from numba import cuda, int64


@cuda.jit
def fill_array(a, multiplier, offset):
    for i in range(cuda.grid(1), a.size, cuda.gridsize(1)):
        a[i] = (multiplier * i + offset) % 11 - 5


@cuda.jit(device=True)
def add_device(a, b):
    return a + b


@cuda.jit(device=True)
def subtract_device(a, b):
    return a - b


@cuda.jit(device=True)
def multiply_device(a, b):
    return a * b


@cuda.jit
def add_kernel(a, b, c):
    for i in range(cuda.grid(1), c.size, cuda.gridsize(1)):
        c[i] = add_device(a[i], b[i])


@cuda.jit
def subtract_kernel(a, b, c):
    for i in range(cuda.grid(1), c.size, cuda.gridsize(1)):
        c[i] = subtract_device(a[i], b[i])


@cuda.jit
def multiply_kernel(a, b, c):
    for i in range(cuda.grid(1), c.size, cuda.gridsize(1)):
        c[i] = multiply_device(a[i], b[i])


def add_cpu(a, b, c):
    for i in range(c.size):
        c[i] = a[i] + b[i]


def subtract_cpu(a, b, c):
    for i in range(c.size):
        c[i] = a[i] - b[i]


def multiply_cpu(a, b, c):
    for i in range(c.size):
        c[i] = a[i] * b[i]


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
        cuda.syncthreads()  # Every thread must reach each barrier.
        step *= 2
    if tid == 0:
        partials[cuda.blockIdx.x] = shared[0]


def verify_result(cpu, gpu, total):
    return np.array_equal(cpu, gpu) and int(np.sum(cpu, dtype=np.int64)) == total


def run_case(n, blocks, threads):
    stream_a, stream_b = cuda.stream(), cuda.stream()
    a, b = cuda.pinned_array(n, np.int32), cuda.pinned_array(n, np.int32)
    d_a = cuda.device_array(n, np.int32, stream=stream_a)
    d_b = cuda.device_array(n, np.int32, stream=stream_b)
    d_c = cuda.device_array(n, np.int32, stream=stream_a)
    partials = cuda.device_array(blocks, np.int64, stream=stream_a)
    d_total = cuda.device_array(1, np.int64, stream=stream_a)
    total = cuda.pinned_array(1, np.int64)
    shared_bytes = threads * np.dtype(np.int64).itemsize

    fill_array[blocks, threads, stream_a](d_a, 3, 1)
    fill_array[blocks, threads, stream_b](d_b, 7, 4)
    d_a.copy_to_host(a, stream=stream_a)
    d_b.copy_to_host(b, stream=stream_b)
    stream_a.synchronize()
    stream_b.synchronize()
    if n == 10:
        print("A =", a)
        print("B =", b)

    correct = True
    for name, host, kernel in (
        ("addition", add_cpu, add_kernel),
        ("subtraction", subtract_cpu, subtract_kernel),
        ("multiplication", multiply_cpu, multiply_kernel),
    ):
        cpu_c = np.empty(n, np.int32)
        gpu_c = cuda.pinned_array(n, np.int32)
        start = perf_counter()
        host(a, b, cpu_c)
        cpu_ms = (perf_counter() - start) * 1000

        kernel[blocks, threads, stream_a](d_a, d_b, d_c)  # Warm-up.
        stream_a.synchronize()
        start = perf_counter()
        kernel[blocks, threads, stream_a](d_a, d_b, d_c)
        stream_a.synchronize()
        gpu_ms = (perf_counter() - start) * 1000

        reduce_sum[blocks, threads, stream_a, shared_bytes](d_c, partials)
        reduce_sum[1, threads, stream_a, shared_bytes](partials, d_total)
        d_c.copy_to_host(gpu_c, stream=stream_a)
        d_total.copy_to_host(total, stream=stream_a)
        stream_a.synchronize()
        correct = verify_result(cpu_c, gpu_c, int(total[0])) and correct
        if n == 10:
            print(f"CPU C ({name}) =", cpu_c)
            print(f"GPU C ({name}) =", gpu_c)
            print(f"CPU/GPU sum ({name}): {np.sum(cpu_c)} / {total[0]}")
        if name == "multiplication":
            print(f"N={n}: CPU multiplication {cpu_ms:.3f} ms; GPU {gpu_ms:.3f} ms")
    print("Verification:", "PASS" if correct else "FAIL")
    return correct


def main():
    device = cuda.get_current_device()
    threads = 256  # Course default: eight warps and a power of two.
    blocks = device.WARP_SIZE * device.MULTIPROCESSOR_COUNT
    print("Hardware name:", device.name)
    print(f"Blocks: {blocks}; threads per block: {threads}")
    try:
        run_case(10, blocks, threads)
        run_case(2 << 20, blocks, threads)
    finally:
        cuda.close()  # Local buffers are released; close any deferred allocations.


if __name__ == "__main__":
    main()
