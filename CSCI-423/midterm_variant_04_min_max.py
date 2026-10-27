"""Task: find the minimum and maximum of a signed integer array A.

For N=10 and N=2<<20, fill A in [-5, 5] on the GPU with an index rule
in a CUDA stream and copy it asynchronously to a pinned host array.
Use grid-stride loops and shared-memory minimum/maximum reductions with
neutral values for idle threads. Print small inputs and both extrema,
verify against a plain Python loop, and time the complete calculation.
Use 256 threads and WARP_SIZE * SM_COUNT blocks. GPU time includes launch/
synchronization but excludes warm-up and copies; CPU time includes Python
overhead. Release the CUDA context at exit.
"""

from time import perf_counter
import numpy as np
from numba import cuda, int64


@cuda.jit
def fill_array(a, multiplier, offset):
    for i in range(cuda.grid(1), a.size, cuda.gridsize(1)):
        a[i] = (multiplier * i + offset) % 11 - 5


@cuda.jit
def reduce_min_max(min_values, max_values, partial_min, partial_max):
    shared = cuda.shared.array(0, dtype=int64)
    tid = cuda.threadIdx.x
    width = cuda.blockDim.x
    minimum = int64(9223372036854775807)
    maximum = int64(-9223372036854775808)
    for i in range(cuda.grid(1), min_values.size, cuda.gridsize(1)):
        minimum = min(minimum, min_values[i])
        maximum = max(maximum, max_values[i])
    # Idle threads keep neutral extrema, so padding cannot change the result.
    shared[tid] = minimum
    shared[width + tid] = maximum
    cuda.syncthreads()

    # The reduction tree requires a power-of-two block size.
    stride = 1
    while stride < width:
        if tid % (2 * stride) == 0:
            shared[tid] = min(shared[tid], shared[tid + stride])
            shared[width + tid] = max(shared[width + tid], shared[width + tid + stride])
        cuda.syncthreads()  # Every thread must reach every barrier.
        stride *= 2
    if tid == 0:
        partial_min[cuda.blockIdx.x] = shared[0]
        partial_max[cuda.blockIdx.x] = shared[width]


def min_max_cpu(a):
    minimum = maximum = int(a[0])
    for i in range(1, a.size):
        value = int(a[i])
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    return minimum, maximum


def verify(cpu, gpu):
    passed = cpu == gpu
    print("Verification:", "PASS" if passed else "FAIL")
    return passed


def run_case(n, blocks, threads):
    stream = cuda.stream()
    a = cuda.pinned_array(n, dtype=np.int32)
    host_min = cuda.pinned_array(1, dtype=np.int64)
    host_max = cuda.pinned_array(1, dtype=np.int64)
    d_a = cuda.device_array(n, dtype=np.int32, stream=stream)
    partial_min = cuda.device_array(blocks, dtype=np.int64, stream=stream)
    partial_max = cuda.device_array(blocks, dtype=np.int64, stream=stream)
    minimum = cuda.device_array(1, dtype=np.int64, stream=stream)
    maximum = cuda.device_array(1, dtype=np.int64, stream=stream)
    shared_bytes = 2 * threads * np.dtype(np.int64).itemsize

    fill_array[blocks, threads, stream](d_a, 3, 1)
    d_a.copy_to_host(a, stream=stream)
    stream.synchronize()  # Complete the pinned copy before host access.

    start = perf_counter()
    cpu = min_max_cpu(a)
    cpu_ms = (perf_counter() - start) * 1000

    def calculate():
        reduce_min_max[blocks, threads, stream, shared_bytes](
            d_a, d_a, partial_min, partial_max)
        # A kernel boundary synchronizes all first-pass blocks.
        reduce_min_max[1, threads, stream, shared_bytes](
            partial_min, partial_max, minimum, maximum)

    calculate()  # Compile and warm up both reduction passes.
    stream.synchronize()
    start = perf_counter()
    calculate()
    stream.synchronize()
    gpu_ms = (perf_counter() - start) * 1000
    minimum.copy_to_host(host_min, stream=stream)
    maximum.copy_to_host(host_max, stream=stream)
    stream.synchronize()
    gpu = int(host_min[0]), int(host_max[0])

    print(f"\nN = {n:,}")
    if n <= 10:
        print("A =", a)
        print(f"Minimum: CPU = {cpu[0]}, GPU = {gpu[0]}")
        print(f"Maximum: CPU = {cpu[1]}, GPU = {gpu[1]}")
    passed = verify(cpu, gpu)
    print(f"CPU minimum/maximum (Python): {cpu_ms:.6f} ms")
    print(f"GPU minimum/maximum reduction: {gpu_ms:.6f} ms")
    return passed  # Local buffers and streams are released on return.


def main():
    try:
        device = cuda.get_current_device()
        name = device.name.decode() if isinstance(device.name, bytes) else device.name
        threads = 256
        blocks = device.WARP_SIZE * device.MULTIPROCESSOR_COUNT
        print(f"GPU: {name}; blocks: {blocks}; threads per block: {threads}")
        run_case(10, blocks, threads)
        run_case(2 << 20, blocks, threads)
    finally:
        cuda.close()


if __name__ == "__main__":
    main()
