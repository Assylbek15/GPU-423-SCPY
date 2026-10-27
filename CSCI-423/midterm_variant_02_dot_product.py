"""Task: compute dot(A, B) = sum(A[i] * B[i]) on CPU and GPU.

For N=10 and N=2<<20, fill signed integer inputs in [-5, 5] on the GPU
using index rules in independent streams; copy them to pinned host arrays.
Use grid-stride loops, int64 products, and a shared-memory sum reduction.
Print the small arrays and dot product, verify against a plain Python loop,
and time the complete calculation. Use 256 threads and WARP_SIZE * SM_COUNT
blocks. GPU time includes launch/synchronization but excludes warm-up and
copies; CPU time includes Python overhead. Release the CUDA context at exit.
"""

from time import perf_counter
import numpy as np
from numba import cuda, int64


@cuda.jit
def fill_array(a, multiplier, offset):
    for i in range(cuda.grid(1), a.size, cuda.gridsize(1)):
        a[i] = (multiplier * i + offset) % 11 - 5


@cuda.jit
def multiply_kernel(a, b, products):
    for i in range(cuda.grid(1), products.size, cuda.gridsize(1)):
        products[i] = int64(a[i]) * int64(b[i])


@cuda.jit
def reduce_sum(values, partial):
    shared = cuda.shared.array(0, dtype=int64)
    tid = cuda.threadIdx.x
    subtotal = int64(0)
    for i in range(cuda.grid(1), values.size, cuda.gridsize(1)):
        subtotal += values[i]
    shared[tid] = subtotal  # Idle threads contribute zero.
    cuda.syncthreads()

    # The reduction tree requires a power-of-two block size.
    stride = 1
    while stride < cuda.blockDim.x:
        if tid % (2 * stride) == 0:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()  # Every thread must reach every barrier.
        stride *= 2
    if tid == 0:
        partial[cuda.blockIdx.x] = shared[0]


def dot_cpu(a, b):
    total = 0
    for i in range(a.size):
        total += int(a[i]) * int(b[i])
    return total


def verify(cpu, gpu):
    passed = cpu == gpu
    print("Verification:", "PASS" if passed else "FAIL")
    return passed


def run_case(n, blocks, threads):
    stream_a, stream_b = cuda.stream(), cuda.stream()
    a = cuda.pinned_array(n, dtype=np.int32)
    b = cuda.pinned_array(n, dtype=np.int32)
    result = cuda.pinned_array(1, dtype=np.int64)
    d_a = cuda.device_array(n, dtype=np.int32, stream=stream_a)
    d_b = cuda.device_array(n, dtype=np.int32, stream=stream_b)
    products = cuda.device_array(n, dtype=np.int64, stream=stream_a)
    partial = cuda.device_array(blocks, dtype=np.int64, stream=stream_a)
    total = cuda.device_array(1, dtype=np.int64, stream=stream_a)
    shared_bytes = threads * np.dtype(np.int64).itemsize

    fill_array[blocks, threads, stream_a](d_a, 3, 1)
    fill_array[blocks, threads, stream_b](d_b, 7, 4)
    d_a.copy_to_host(a, stream=stream_a)
    d_b.copy_to_host(b, stream=stream_b)
    stream_a.synchronize()
    stream_b.synchronize()  # Both inputs are ready before either is consumed.

    start = perf_counter()
    cpu = dot_cpu(a, b)
    cpu_ms = (perf_counter() - start) * 1000

    def calculate():
        multiply_kernel[blocks, threads, stream_a](d_a, d_b, products)
        reduce_sum[blocks, threads, stream_a, shared_bytes](products, partial)
        # The final block uses grid-stride loading even if blocks > threads.
        reduce_sum[1, threads, stream_a, shared_bytes](partial, total)

    calculate()  # Compile and warm up all computation kernels.
    stream_a.synchronize()
    start = perf_counter()
    calculate()
    stream_a.synchronize()
    gpu_ms = (perf_counter() - start) * 1000
    total.copy_to_host(result, stream=stream_a)
    stream_a.synchronize()
    gpu = int(result[0])

    print(f"\nN = {n:,}")
    if n <= 10:
        host_products = cuda.pinned_array(n, dtype=np.int64)
        products.copy_to_host(host_products, stream=stream_a)
        stream_a.synchronize()
        print("A =", a)
        print("B =", b)
        print("A * B =", host_products)
        print(f"Dot product: CPU = {cpu}, GPU = {gpu}")
    passed = verify(cpu, gpu)
    print(f"CPU dot product (Python): {cpu_ms:.6f} ms")
    print(f"GPU products + reduction: {gpu_ms:.6f} ms")
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
