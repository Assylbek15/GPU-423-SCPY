"""Shared-memory reduction. Run on a CUDA GPU: python reduce0.py.

Each block produces one partial sum. Repeat until only one sum remains.
"""

import numpy as np
from numba import cuda, int64


@cuda.jit
def reduce0(g_idata, g_odata):
    sdata = cuda.shared.array(shape=0, dtype=int64)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    # Zero padding makes a partly occupied last block safe.
    sdata[tid] = g_idata[i] if i < g_idata.size else 0
    cuda.syncthreads()

    s = 1
    while s < cuda.blockDim.x:
        if tid % (2 * s) == 0:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s *= 2

    if tid == 0:
        g_odata[cuda.blockIdx.x] = sdata[0]


def sum_gpu(values, threads_per_block=256):
    threads = threads_per_block
    # The reduction tree requires a power-of-two block size.
    if threads < 2 or threads > 1024 or threads & (threads - 1):
        raise ValueError("threads must be a power of two between 2 and 1024")
    if values.size == 0:
        return 0

    current = cuda.to_device(values)
    shared_bytes = threads * np.dtype(np.int64).itemsize
    # Keep intermediates alive until the final synchronous copy completes.
    buffers = [current]
    while current.size > 1:
        blocks = (current.size + threads - 1) // threads
        partial = cuda.device_array(blocks, dtype=np.int64)
        # Numba launch order: [blocks, threads, stream, shared_memory_bytes].
        reduce0[blocks, threads, 0, shared_bytes](current, partial)
        buffers.append(partial)
        current = partial

    return int(current.copy_to_host()[0])


def main():
    n = 1024
    threads = 256
    a = np.arange(n, dtype=np.int32)
    try:
        gpu_sum = sum_gpu(a, threads)
        cpu_sum = int(np.sum(a, dtype=np.int64))
        print(f"GPU SUM: {gpu_sum}")
        print(f"CPU SUM: {cpu_sum}")
        if gpu_sum != cpu_sum:
            raise AssertionError(f"Reduction mismatch: {gpu_sum} != {cpu_sum}")
        print("Verification: PASS")
        # Expected: 0 + 1 + ... + 1023 = 523776.
    finally:
        cuda.close()


if __name__ == "__main__":
    main()
