'''
implement reduction algorithm in python
'''
from numba import cuda, types as numba_types
import numpy as np


@cuda.jit
def reduce0(g_idata, g_odata):
  sdata = cuda.shared.array(shape=0, dtype=numba_types.int32)

  tid = cuda.threadIdx.x
  i = cuda.grid(1)

  sdata[tid] = g_idata[i]
  cuda.syncthreads()

  s = 1
  while s < cuda.blockDim.x:
    if tid % (2 * s) == 0:
      sdata[tid] += sdata[tid + s]
    cuda.syncthreads()
    s *= 2

  if tid == 0:
    g_odata[cuda.blockIdx.x] = sdata[0]
  

n = 1024
threads_per_block = 1024
blocks = (threads_per_block + n - 1) // threads_per_block

shared_mem_bytes1 = threads_per_block * np.dtype(np.int32).itemsize
shared_mem_bytes2 = blocks * np.dtype(np.int32).itemsize

a = np.arange(n).astype(np.int32)
out = np.zeros_like(a).astype(np.int32)

d_a = cuda.to_device(a)
d_out_1 = cuda.to_device(out)

out_2 = np.array([1])
d_out_2 = cuda.to_device(out_2)

reduce0[blocks, threads_per_block, 0, shared_mem_bytes1](d_a, d_out_1)
reduce0[1, blocks, 0, shared_mem_bytes2](d_out_1, d_out_2)

sum_gpu = d_out_2.copy_to_host()
sum_cpu = np.sum(a)

print(f"GPU SUM: {sum_gpu[:blocks+2]}")
print(f"GPU RESULT: {np.sum(sum_gpu)}")
print(f"CPU SUM: {sum_cpu}")