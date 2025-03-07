import numpy as np
import pandas as pd


def my_dft(x_array)->np.ndarray:
    """
    自己编写一个DFT程序，要求接受元素类型为
    complex128(python)/ComplexF64(julia) 
    的⼀维数组，返回其离散傅⾥叶变换
    """
    N = len(x_array)
    n = np.arange(N)
    k = n.reshape((N,1))
    M = np.exp(-2j * np.pi * k * n / N)/N**0.5  #np.exp(np.array) 会对数组中的每个元素应用指数函数 exp
    return np.dot(M,x_array)

# 测试样例:随机数组
np.random.seed(0)  # 设置seed可以保证输出结果相同
x = np.random.rand(8) + 1j * np.random.rand(8)
x = x.astype(np.complex128)
# Dataframe.astype(dtype, copy:defalt True, errors='raise')Cast a pandas object to a specified dtype dtype.

my_dft_result = my_dft(x)
FFTW_result = np.fft.fft(x,norm='ortho')

print("\nOriginal data",x)
print("My DFT result",my_dft_result)
print("\nNumpy FFTW result",FFTW_result)

print("\nMy DFT result - Numpy FFTW result",my_dft_result-FFTW_result)




