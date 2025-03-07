# 结果及讨论
## A.DFT和FFT
### 1.编写一个DFT程序并验证

如下所示，编写`my_dft`函数，实现对输入的数组进行DFT计算，并返回结果。
````python
def my_dft(x_array)->np.ndarray:
    N = len(x_array)
    n = np.arange(N)
    k = n.reshape((N,1))
    M = np.exp(-2j * np.pi * k * n / N)/N**0.5  
    return np.dot(M,x_array)
````
使用一个八维的随机复向量作为测试，并且使用`np.fft`作为标准包函数，测试数据、`my_dft`、`np.fft`给出的结果以及二者差值如下：

![](figure/A_1.png)

由上图结果可以看出，自己编写的函数和包函数给出的结果差值极小，可以认为自己编写的dft函数具有好的离散傅里叶变换的功能。