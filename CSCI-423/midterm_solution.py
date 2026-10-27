"""Midterm solution. Run on a CUDA GPU: python midterm_solution.py

Requires numpy and numba (or numba with the numba-cuda package).
N=10 displays results; N=2<<20 measures multiplication time.
CPU functions are ordinary Python loops, so their timing includes interpreter
overhead. The GPU timing excludes compilation, transfers, and reduction.
"""

from time import perf_counter
import numpy as np
from numba import cuda, int64


@cuda.jit
def fill_array(a, multiplier, offset):
    # The exam permits an index-based rule instead of a random generator.
    for i in range(cuda.grid(1), a.size, cuda.gridsize(1)):
        a[i] = (multiplier * i + offset) % 11 - 5


# Device functions run inside kernels; they are not launched from the host.
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
def reduce0(values, partial_sums):
    """One int64 sum per block, with grid-stride loading before the tree."""
    shared = cuda.shared.array(shape=0, dtype=int64)
    tid = cuda.threadIdx.x
    subtotal = int64(0)
    for i in range(cuda.grid(1), values.size, cuda.gridsize(1)):
        subtotal += values[i]
    shared[tid] = subtotal  # Empty threads contribute zero.
    cuda.syncthreads()

    stride = 1
    while stride < cuda.blockDim.x:
        if tid % (2 * stride) == 0:
            shared[tid] += shared[tid + stride]
        # This barrier MUST be outside the if: every thread participates.
        cuda.syncthreads()
        stride *= 2

    if tid == 0:
        partial_sums[cuda.blockIdx.x] = shared[0]


def sum_gpu(d_values, blocks, threads, stream):
    """Reduce device values with a full first grid and one final block."""
    shared_bytes = threads * np.dtype(np.int64).itemsize
    partial = cuda.device_array(blocks, dtype=np.int64, stream=stream)
    total = cuda.device_array(1, dtype=np.int64, stream=stream)
    host_total = cuda.pinned_array(1, dtype=np.int64)
    reduce0[blocks, threads, stream, shared_bytes](d_values, partial)
    # A kernel boundary provides grid-wide synchronization. Grid-stride loading
    # lets this ONE block consume any number of partial sums, even > threads.
    reduce0[1, threads, stream, shared_bytes](partial, total)
    total.copy_to_host(host_total, stream=stream)
    stream.synchronize()  # Before reading host_total or releasing buffers.
    return int(host_total[0])


def verify_result(cpu_result, gpu_result, gpu_sum):
    """Verify every element AND the reduction, not just a printed sample."""
    np.testing.assert_array_equal(gpu_result, cpu_result)
    expected_sum = int(np.sum(cpu_result, dtype=np.int64))
    if gpu_sum != expected_sum:
        raise AssertionError(f"Sum mismatch: GPU {gpu_sum}, CPU {expected_sum}")


def time_gpu_ms(d_a, d_b, d_c, blocks, threads, stream):
    # Warm-up compiles the kernel and finishes before timing begins.
    multiply_kernel[blocks, threads, stream](d_a, d_b, d_c)
    stream.synchronize()
    start, stop = cuda.event(timing=True), cuda.event(timing=True)
    start.record(stream)
    multiply_kernel[blocks, threads, stream](d_a, d_b, d_c)
    stop.record(stream)
    stop.synchronize()
    return cuda.event_elapsed_time(start, stop)


def run_case(n, blocks, threads):
    stream_a, stream_b = cuda.stream(), cuda.stream()
    a = cuda.pinned_array(n, dtype=np.int32)
    b = cuda.pinned_array(n, dtype=np.int32)
    d_a = cuda.device_array(n, dtype=np.int32, stream=stream_a)
    d_b = cuda.device_array(n, dtype=np.int32, stream=stream_b)
    d_c = cuda.device_array(n, dtype=np.int32, stream=stream_a)

    # Launch both independent fills before waiting for either one.
    fill_array[blocks, threads, stream_a](d_a, 3, 1)
    fill_array[blocks, threads, stream_b](d_b, 7, 4)
    d_a.copy_to_host(a, stream=stream_a)
    d_b.copy_to_host(b, stream=stream_b)
    # Wait for both inputs and their asynchronous copies before using them.
    stream_a.synchronize()
    stream_b.synchronize()

    operations = (
        ("addition", add_cpu, add_kernel),
        ("subtraction", subtract_cpu, subtract_kernel),
        ("multiplication", multiply_cpu, multiply_kernel),
    )
    cpu_results, gpu_results, cpu_sums, gpu_sums = {}, {}, {}, {}
    for name, host_function, kernel in operations:
        cpu_c = np.empty(n, dtype=np.int32)
        gpu_c = cuda.pinned_array(n, dtype=np.int32)
        host_function(a, b, cpu_c)
        kernel[blocks, threads, stream_a](d_a, d_b, d_c)
        d_c.copy_to_host(gpu_c, stream=stream_a)  # Asynchronous, pinned D->H.
        total = sum_gpu(d_c, blocks, threads, stream_a)
        # sum_gpu synchronizes this stream, so gpu_c is now ready as well.
        verify_result(cpu_c, gpu_c, total)
        cpu_results[name], gpu_results[name] = cpu_c, gpu_c
        cpu_sums[name] = int(np.sum(cpu_c, dtype=np.int64))
        gpu_sums[name] = total

    if n == 10:
        print(f"\nGenerated arrays when N = {n}:")
        print("A =", a)
        print("B =", b)
        print("CPU results:")
        for name, _, _ in operations:
            print(f"C ({name}) =", cpu_results[name])
        for name, _, _ in operations:
            print(f"Sum ({name}) = {cpu_sums[name]}")
        print("GPU results:")
        for name, _, _ in operations:
            print(f"C ({name}) =", gpu_results[name])
        for name, _, _ in operations:
            print(f"Sum ({name}) = {gpu_sums[name]}")
    print(f"Verification for N = {n}: PASS (all elements and all three sums)")

    if n == 2 << 20:
        cpu_c = cpu_results["multiplication"]
        start = perf_counter()
        multiply_cpu(a, b, cpu_c)
        cpu_ms = (perf_counter() - start) * 1000
        gpu_ms = time_gpu_ms(d_a, d_b, d_c, blocks, threads, stream_a)
        print(f"\nTime to complete when N = 2<<20 ({n:,} elements):")
        print(f"CPU multiplication (Python): {cpu_ms:.6f} ms")
        print(f"GPU multiplication:         {gpu_ms:.6f} ms")

    # Local array/stream references are released when this function returns.


def main():
    try:
        device = cuda.get_current_device()
        name = device.name.decode() if isinstance(device.name, bytes) else device.name
        # Course choice: 256 = 8 warps, and a power of two for reduction.
        # This is a practical default, not a measured optimum for every GPU.
        threads = 256
        blocks = device.WARP_SIZE * device.MULTIPROCESSOR_COUNT
        print(f"Hardware name: {name}")
        print(f"Threads per block: {threads}; blocks: {blocks}")
        run_case(10, blocks, threads)
        run_case(2 << 20, blocks, threads)
    finally:
        # Release the CUDA context and any deferred device allocations.
        cuda.close()


if __name__ == "__main__":
    main()
