"""Practice 09: for NxN A, produce row[i] = sum_j A[i,j] and col[j] = sum_i A[i,j].

Use N=4 to display A and both CPU/GPU vectors; repeat with N=1<<6 for timings.
Initialize integer values in [-5,5] on a stream and copy to pinned host memory.
Use grid-stride kernels, 256 threads, and WARP_SIZE*SM_count blocks.
Verify both int64 vectors against plain Python loops and release CUDA resources.
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
def row_column_sums_kernel(a, rows, columns):
    n = a.shape[0]
    # A thread owns one output pair, so no atomics or shared-memory tree is needed.
    for i in range(cuda.grid(1), n, cuda.gridsize(1)):
        row_total = int64(0)
        column_total = int64(0)
        for j in range(n):
            row_total += int64(a[i, j])
            column_total += int64(a[j, i])
        rows[i] = row_total
        columns[i] = column_total


def row_column_sums_cpu(a, rows, columns):
    n = a.shape[0]
    for i in range(n):
        row_total = 0
        column_total = 0
        for j in range(n):
            row_total += int(a[i, j])
            column_total += int(a[j, i])
        rows[i] = row_total
        columns[i] = column_total


def verify_result(cpu_rows, cpu_columns, gpu_rows, gpu_columns):
    passed = bool(np.array_equal(cpu_rows, gpu_rows)
                  and np.array_equal(cpu_columns, gpu_columns))
    print("Verification:", "PASS" if passed else "FAIL")
    return passed


def run_case(n, blocks, threads):
    stream = cuda.stream()
    a = cuda.pinned_array((n, n), dtype=np.int32)
    gpu_rows = cuda.pinned_array(n, dtype=np.int64)
    gpu_columns = cuda.pinned_array(n, dtype=np.int64)
    cpu_rows = np.empty(n, dtype=np.int64)
    cpu_columns = np.empty(n, dtype=np.int64)
    d_a = cuda.device_array((n, n), dtype=np.int32, stream=stream)
    d_rows = cuda.device_array(n, dtype=np.int64, stream=stream)
    d_columns = cuda.device_array(n, dtype=np.int64, stream=stream)

    fill_matrix[blocks, threads, stream](d_a, 3, 1)
    d_a.copy_to_host(a, stream=stream)
    stream.synchronize()  # Finish the asynchronous copy before CPU reads.

    start = perf_counter()
    row_column_sums_cpu(a, cpu_rows, cpu_columns)
    cpu_ms = (perf_counter() - start) * 1000

    row_column_sums_kernel[blocks, threads, stream](d_a, d_rows, d_columns)
    stream.synchronize()  # Compile and warm up before timing.
    start = perf_counter()
    row_column_sums_kernel[blocks, threads, stream](d_a, d_rows, d_columns)
    stream.synchronize()
    gpu_ms = (perf_counter() - start) * 1000
    d_rows.copy_to_host(gpu_rows, stream=stream)
    d_columns.copy_to_host(gpu_columns, stream=stream)
    stream.synchronize()

    print(f"\nN = {n}")
    if n == 4:
        print("A =\n", a)
        print("CPU row sums =", cpu_rows)
        print("GPU row sums =", gpu_rows)
        print("CPU column sums =", cpu_columns)
        print("GPU column sums =", gpu_columns)
    passed = verify_result(cpu_rows, cpu_columns, gpu_rows, gpu_columns)
    print(f"CPU row and column sums (Python): {cpu_ms:.6f} ms")
    print(f"GPU row and column sums:          {gpu_ms:.6f} ms")
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
