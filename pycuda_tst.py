import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
a = np.zeros()
b = np.array([[5,6],[7,8]]).astype(np.float32)
# 定义CUDA kernel代码
mod = SourceModule("""
    __global__ void matrix_multiply(float *a, float *b, float *c)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        c[idx] = a[idx] * b[idx];
    }
""" % {"MATRIX_SIZE": 2})
 
# 调用CUDA kernel函数
matrix_multiply = mod.get_function("matrix_multiply")
 
# 定义输出数组
c = np.zeros([2,2]).astype(np.float32)
 
MATRIX_SIZE = np.int32(2)
block_size = (16, 16, 1)
grid_size = (int(np.ceil(2/block_size[0])), int(np.ceil(2/block_size[1])), 1)
 
matrix_multiply(drv.In(a), drv.In(b), drv.Out(c), grid=grid_size, block=block_size)
print(c)