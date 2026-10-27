"""Practice 12: remove shared-memory bank conflicts from matrix transpose.

Transpose the same NxN int32 matrix using tiles shaped (32,32) and (32,33).
Both kernels coalesce global accesses; the extra column avoids the 32-way
shared-memory bank conflict when a warp reads a column of 32-bit values.
Verify both against a CPU transpose and compare CUDA-event kernel timings.
Use N=32 and N=1024, (32,8) threads, and WARP_SIZE*SM_count blocks. N must
be a multiple of 32. Display the top-left 4x4 corner for the small case.
Use pinned copies, exclude warm-up/transfers from GPU timing, and clear memory.
Timings depend on the GPU; padding changes storage layout, not the result.
"""

from time import perf_counter
import numpy as np
from numba import cuda, int32

TILE = 32


@cuda.jit
def fill_matrix(a):
    n = a.shape[1]
    for i in range(cuda.grid(1), a.size, cuda.gridsize(1)):
        row, col = i // n, i % n
        a[row, col] = (3 * row + 7 * col + 1) % 11 - 5


@cuda.jit
def transpose_unpadded(a, out):
    tile = cuda.shared.array((TILE, TILE), dtype=int32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tiles_per_row = a.shape[0] // TILE
    for tile_id in range(cuda.blockIdx.x, tiles_per_row ** 2, cuda.gridDim.x):
        tile_row, tile_col = tile_id // tiles_per_row, tile_id % tiles_per_row
        x, y = tile_col * TILE + tx, tile_row * TILE + ty
        for j in range(0, TILE, cuda.blockDim.y):
            tile[ty + j, tx] = a[y + j, x]
        cuda.syncthreads()

        out_x, out_y = tile_row * TILE + tx, tile_col * TILE + ty
        for j in range(0, TILE, cuda.blockDim.y):
            out[out_y + j, out_x] = tile[tx, ty + j]
        cuda.syncthreads()  # Protect shared storage before the next tile.


@cuda.jit
def transpose_padded(a, out):
    # Only the allocation changes: column reads now visit distinct banks.
    tile = cuda.shared.array((TILE, TILE + 1), dtype=int32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tiles_per_row = a.shape[0] // TILE
    for tile_id in range(cuda.blockIdx.x, tiles_per_row ** 2, cuda.gridDim.x):
        tile_row, tile_col = tile_id // tiles_per_row, tile_id % tiles_per_row
        x, y = tile_col * TILE + tx, tile_row * TILE + ty
        for j in range(0, TILE, cuda.blockDim.y):
            tile[ty + j, tx] = a[y + j, x]
        cuda.syncthreads()

        out_x, out_y = tile_row * TILE + tx, tile_col * TILE + ty
        for j in range(0, TILE, cuda.blockDim.y):
            out[out_y + j, out_x] = tile[tx, ty + j]
        cuda.syncthreads()


def transpose_cpu(a, out):
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            out[col, row] = a[row, col]


def verify_result(cpu, gpu):
    correct = np.array_equal(cpu, gpu)
    print("Verification:", "PASS" if correct else "FAIL")
    return correct


def run_case(n, blocks, threads):
    stream = cuda.stream()
    a = cuda.pinned_array((n, n), dtype=np.int32)
    cpu = np.empty((n, n), dtype=np.int32)
    gpu = cuda.pinned_array((n, n), dtype=np.int32)
    d_a = cuda.device_array((n, n), dtype=np.int32, stream=stream)
    d_out = cuda.device_array((n, n), dtype=np.int32, stream=stream)
    fill_matrix[blocks, threads[0] * threads[1], stream](d_a)
    d_a.copy_to_host(a, stream=stream)
    stream.synchronize()

    start_cpu = perf_counter()
    transpose_cpu(a, cpu)
    cpu_ms = (perf_counter() - start_cpu) * 1000
    print(f"\nN={n}; CPU transpose: {cpu_ms:.6f} ms")
    if n == 32:
        print("A (top-left 4x4) =\n", a[:4, :4])
        print("CPU A.T (top-left 4x4) =\n", cpu[:4, :4])

    correct = True
    for name, kernel in (("Unpadded", transpose_unpadded), ("Padded", transpose_padded)):
        kernel[blocks, threads, stream](d_a, d_out)
        stream.synchronize()  # Compile and warm up each version separately.
        start, stop = cuda.event(), cuda.event()
        start.record(stream)
        kernel[blocks, threads, stream](d_a, d_out)
        stop.record(stream)
        stop.synchronize()
        gpu_ms = start.elapsed_time(stop)
        d_out.copy_to_host(gpu, stream=stream)
        stream.synchronize()
        print(f"{name} GPU transpose: {gpu_ms:.6f} ms")
        if n == 32:
            print(f"{name} A.T (top-left 4x4) =\n", gpu[:4, :4])
        correct = verify_result(cpu, gpu) and correct
    return correct


def main():
    try:
        device = cuda.get_current_device()
        blocks = device.WARP_SIZE * device.MULTIPROCESSOR_COUNT
        threads = (TILE, 8)
        print("GPU:", device.name)
        print(f"Blocks: {blocks}; threads per block: {threads} (256 total)")
        run_case(32, blocks, threads)
        run_case(1024, blocks, threads)
    finally:
        cuda.close()


if __name__ == "__main__":
    main()
