"""Practice 13: compare separate row-sum and column-sum kernels for row-major A.

For an NxN int32 matrix, compute rows[i] = sum_j A[i,j] and cols[i] = sum_j A[j,i].
At fixed j, adjacent GPU threads access A[i,j] with stride N in the row kernel
(uncoalesced), versus contiguous A[j,i] in the column kernel (coalesced).
Hardware and caching affect timings; coalescing does not guarantee a faster run.
Use N=4 to display results and N=1024 to compare timings. Initialize values in
[-5,5] on a stream, use pinned asynchronous copies, and verify both int64 outputs
against plain Python loops. Use grid-stride loops, 256 threads per block, and
WARP_SIZE*SM_count blocks. Release CUDA resources on exit.
GPU CUDA-event times exclude compilation and transfers; CPU times include Python
overhead. Requires NumPy, Numba, and a CUDA GPU.
"""

from time import perf_counter
import numpy as np
from numba import cuda, int64


@cuda.jit
def fill_matrix(a):
    n = a.shape[1]
    for index in range(cuda.grid(1), a.size, cuda.gridsize(1)):
        row, col = index // n, index % n
        a[row, col] = (3 * row + 7 * col + 1) % 11 - 5


@cuda.jit
def sum_rows_kernel(a, rows):
    n = a.shape[0]
    for i in range(cuda.grid(1), n, cuda.gridsize(1)):
        total = int64(0)
        for j in range(n):
            # Across adjacent threads, these addresses are N elements apart.
            total += int64(a[i, j])
        rows[i] = total


@cuda.jit
def sum_columns_kernel(a, columns):
    n = a.shape[0]
    for i in range(cuda.grid(1), n, cuda.gridsize(1)):
        total = int64(0)
        for j in range(n):
            # Adjacent threads read adjacent elements at each fixed j.
            total += int64(a[j, i])
        columns[i] = total


def sum_rows_cpu(a, rows):
    n = a.shape[0]
    for i in range(n):
        total = 0
        for j in range(n):
            total += int(a[i, j])
        rows[i] = total


def sum_columns_cpu(a, columns):
    n = a.shape[0]
    for i in range(n):
        total = 0
        for j in range(n):
            total += int(a[j, i])
        columns[i] = total


def verify_result(cpu_result, gpu_result, label):
    passed = bool(np.array_equal(cpu_result, gpu_result))
    print(f"{label} verification:", "PASS" if passed else "FAIL")
    return passed


def time_gpu_ms(kernel, d_a, d_result, blocks, threads, stream):
    kernel[blocks, threads, stream](d_a, d_result)
    stream.synchronize()  # Finish compilation and warm-up before timing.
    start, end = cuda.event(), cuda.event()
    start.record(stream)
    kernel[blocks, threads, stream](d_a, d_result)
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end)


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

    fill_matrix[blocks, threads, stream](d_a)
    d_a.copy_to_host(a, stream=stream)
    stream.synchronize()  # The pinned input must be ready before CPU reads.

    start = perf_counter()
    sum_rows_cpu(a, cpu_rows)
    cpu_rows_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    sum_columns_cpu(a, cpu_columns)
    cpu_columns_ms = (perf_counter() - start) * 1000

    gpu_rows_ms = time_gpu_ms(sum_rows_kernel, d_a, d_rows, blocks, threads, stream)
    gpu_columns_ms = time_gpu_ms(
        sum_columns_kernel, d_a, d_columns, blocks, threads, stream
    )
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
    rows_passed = verify_result(cpu_rows, gpu_rows, "Row sums")
    columns_passed = verify_result(cpu_columns, gpu_columns, "Column sums")
    print(f"CPU row sums (Python):     {cpu_rows_ms:.6f} ms")
    print(f"CPU column sums (Python):  {cpu_columns_ms:.6f} ms")
    print(f"GPU rows (uncoalesced):    {gpu_rows_ms:.6f} ms")
    print(f"GPU columns (coalesced):   {gpu_columns_ms:.6f} ms")
    # Local buffer and stream references are released on return.
    return rows_passed and columns_passed


def main():
    try:
        device = cuda.get_current_device()
        name = device.name.decode() if isinstance(device.name, bytes) else device.name
        threads = 256
        blocks = device.WARP_SIZE * device.MULTIPROCESSOR_COUNT
        print(f"GPU: {name}")
        print(f"Blocks: {blocks}; threads per block: {threads}")
        run_case(4, blocks, threads)
        run_case(1024, blocks, threads)
    finally:
        cuda.close()  # Release the context and deferred device allocations.


if __name__ == "__main__":
    main()
