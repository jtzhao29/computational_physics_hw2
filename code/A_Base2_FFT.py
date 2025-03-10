import numpy as np
from A_dft import my_dft
import cmath
from timeit import timeit
import matplotlib.pyplot as plt
def is_power_of_two(n)->bool:
    """
    判断一个数是否是2的幂
    """
    return n > 0 and (n & (n - 1)) == 0
    #如果是2的幂次，则则其二进制表示中只有一个比特位为 1

def fft(x):
    """
    快速傅里叶变换,自己写的，递归算法
    """
    n = len(x)
    if n==1:
        return x
    
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
    # 利用离散傅里叶变换的性质:傅里叶变换的结果中前一半和后一半互为共轭
    return [even[k] + T[k] for k in range(n // 2)] + \
           [even[k] - T[k] for k in range(n // 2)]

def iterative_fft(x):
    """
    快速傅里叶变换,自己写的，非递归算法，蝴蝶运算的那个
    """
    n = len(x)
    log_n = int(np.log2(n))

    W = [cmath.exp(-2j * cmath.pi / (1 << i)) for i in range(log_n + 1)]

    result = np.array(x)
    indices = np.arange(n)
    indices = np.bitwise_or.reduce([((indices >> i) & 1) << (log_n-1-i) for i in range(log_n)], axis=0)
    result = result[indices]

    half_size = 1
    for level in range(log_n):
        step = half_size * 2
        for start in range(0, n, step):
            factor = 1
            for i in range(half_size):
                even = result[start + i]
                odd = factor * result[start + half_size + i]
                result[start + i] = even + odd
                result[start + half_size + i] = even - odd
                factor *= W[level + 1]
        half_size = step

    return result


if __name__ == '__main__':
    # # 测试样例:随机数组
    # np.random.seed(0)  # 设置seed可以保证输出结果相同
    # x = np.random.rand(8) + 1j * np.random.rand(8)
    # x = x.astype(np.complex128)
   
    # my_fft_y = fft(x)
    # my_dft_y = my_dft(x)
    # FFT_y = np.fft.fft(x)
    # print("\nTesting data",x)
    # print("\nMy FFT result",my_fft_y)
    # # print("\nMy DFT result",my_dft_y)
    # print("\nFFTW result",FFT_y)

    # 给出数组⼤⼩从2**4到2**12的测试样例
    size = [2**i for i in range(4,13)]
    my_fft = []
    my_iterative_fft = []
    np_fft = []
    for i in size:
        if  is_power_of_two(i):
            x = np.random.rand(i) + 1j * np.random.rand(i)
            x = x.astype(np.complex128)
            exec_time_my_fft = timeit(lambda: fft(x),number=1000)
            exec_time_fft = timeit(lambda: np.fft.fft(x),number=1000)
            exec_time_iterative_fft = timeit(lambda: iterative_fft(x),number=1000)
            my_fft.append(exec_time_my_fft/1000)
            my_iterative_fft.append(exec_time_iterative_fft/1000)
            np_fft.append(exec_time_fft/1000)
            
            print(f"my fft for size{i},exection time:{exec_time_my_fft}")
            # print(f"my iterative_fft for size{i},exection time:{exec_time_iterative_fft}")
            print(f"np.fft for size{i},exection time:{exec_time_fft}")

    nlogn = [i*np.log2(i) for i in size]

    plt.plot(size,my_fft,label="my_fft")
    plt.plot(size,my_iterative_fft,label="my_iterative_fft")
    plt.plot(size,np_fft,label="np_fft")
    plt.plot(size,nlogn,label="nlogn")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('size',fontsize=25)
    plt.ylabel('time',fontsize=25)
    plt.title('FFT execution time',fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()