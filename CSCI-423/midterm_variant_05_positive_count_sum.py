"""Task: count A[i] > 0 and sum only those positive elements.

For N=10 and N=2<<20, fill A in [-5, 5] on the GPU with an index rule
in a CUDA stream and copy it asynchronously to a pinned host array.
Map elements to int64 (positive flag, positive value), then use grid-stride
shared-memory reductions. Print small inputs, mapped arrays, count and sum;
verify against a plain Python loop and time the complete calculation.
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
def positive_kernel(a, counts, sums):
    for i in range(cuda.grid(1), a.size, cuda.gridsize(1)):
        counts[i] = 1 if a[i] > 0 else 0
        sums[i] = a[i] if a[i] > 0 else 0


@cuda.jit
def reduce_count_sum(counts, sums, partial_counts, partial_sums):
    shared = cuda.shared.array(0, dtype=int64)
    tid = cuda.threadIdx.x
    width = cuda.blockDim.x
    count, total = int64(0), int64(0)
    for i in range(cuda.grid(1), counts.size, cuda.gridsize(1)):
        count += counts[i]
        total += sums[i]
    shared[tid] = count  # Idle threads contribute zeros to both reductions.
    shared[width + tid] = total
    cuda.syncthreads()

    # The reduction tree requires a power-of-two block size.
    stride = 1
    while stride < width:
        if tid % (2 * stride) == 0:
            shared[tid] += shared[tid + stride]
            shared[width + tid] += shared[width + tid + stride]
        cuda.syncthreads()  # Every thread must reach every barrier.
        stride *= 2
    if tid == 0:
        partial_counts[cuda.blockIdx.x] = shared[0]
        partial_sums[cuda.blockIdx.x] = shared[width]


def positive_count_sum_cpu(a):
    count, total = 0, 0
    for i in range(a.size):
        value = int(a[i])
        if value > 0:
            count += 1
            total += value
    return count, total


def verify(cpu, gpu):
    passed = cpu == gpu
    print("Verification:", "PASS" if passed else "FAIL")
    return passed


def run_case(n, blocks, threads):
    stream = cuda.stream()
    a = cuda.pinned_array(n, dtype=np.int32)
    host_count = cuda.pinned_array(1, dtype=np.int64)
    host_sum = cuda.pinned_array(1, dtype=np.int64)
    d_a = cuda.device_array(n, dtype=np.int32, stream=stream)
    counts = cuda.device_array(n, dtype=np.int64, stream=stream)
    sums = cuda.device_array(n, dtype=np.int64, stream=stream)
    partial_counts = cuda.device_array(blocks, dtype=np.int64, stream=stream)
    partial_sums = cuda.device_array(blocks, dtype=np.int64, stream=stream)
    count = cuda.device_array(1, dtype=np.int64, stream=stream)
    total = cuda.device_array(1, dtype=np.int64, stream=stream)
    shared_bytes = 2 * threads * np.dtype(np.int64).itemsize

    fill_array[blocks, threads, stream](d_a, 3, 1)
    d_a.copy_to_host(a, stream=stream)
    stream.synchronize()  # Complete the pinned copy before host access.

    start = perf_counter()
    cpu = positive_count_sum_cpu(a)
    cpu_ms = (perf_counter() - start) * 1000

    def calculate():
        positive_kernel[blocks, threads, stream](d_a, counts, sums)
        reduce_count_sum[blocks, threads, stream, shared_bytes](
            counts, sums, partial_counts, partial_sums)
        reduce_count_sum[1, threads, stream, shared_bytes](
            partial_counts, partial_sums, count, total)

    calculate()  # Compile and warm up all computation kernels.
    stream.synchronize()
    start = perf_counter()
    calculate()
    stream.synchronize()
    gpu_ms = (perf_counter() - start) * 1000
    count.copy_to_host(host_count, stream=stream)
    total.copy_to_host(host_sum, stream=stream)
    stream.synchronize()
    gpu = int(host_count[0]), int(host_sum[0])

    print(f"\nN = {n:,}")
    if n <= 10:
        host_counts = cuda.pinned_array(n, dtype=np.int64)
        host_sums = cuda.pinned_array(n, dtype=np.int64)
        counts.copy_to_host(host_counts, stream=stream)
        sums.copy_to_host(host_sums, stream=stream)
        stream.synchronize()
        print("A =", a)
        print("Positive flags =", host_counts)
        print("Positive values =", host_sums)
        print(f"Positive count: CPU = {cpu[0]}, GPU = {gpu[0]}")
        print(f"Positive sum: CPU = {cpu[1]}, GPU = {gpu[1]}")
    passed = verify(cpu, gpu)
    print(f"CPU positive count/sum (Python): {cpu_ms:.6f} ms")
    print(f"GPU mapping + reduction: {gpu_ms:.6f} ms")
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
