import numpy as np
import matplotlib.pyplot as plt

def recursion(x:complex)->complex:
    """
    输入$z_n$,输出$z_{n+1}$
    """
    return (2*x**3 + 1)/(3*x**2)

def f(x:complex)->complex:
    """
    输入$z_0$,输出函数值
    """
    return x**3-1

def newton_method(xo:complex, max_iter=100, tol=1e-6)->complex:
    """
    牛顿法，输入x0为初始值，输出为收敛解
    """
    x = xo
    for i in range (max_iter):
        x = recursion(x)
        if abs(f(x)) < tol:
            break
    return x

def get_convergence_region(center, width, resolution, bias=1e-5):
    """
    计算牛顿法收敛区域
    输入中心，宽度，分辨率，容忍度，
    输出收敛解表格
    """
    x_min, x_max = center - width , center + width 
    y_min, y_max = center - width , center + width 
    
    x_vals = np.linspace(x_min, x_max, int(2*width/resolution))
    y_vals = np.linspace(y_min, y_max, int(2*width/resolution))
    X,Y = np.meshgrid(x_vals, y_vals)
    Z = X + 1j * Y

    roots = [1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)]

    region = np.zeros(Z.shape, dtype=int)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            z = Z[i, j]
            z_final = newton_method(z, max_iter=50, tol=bias)
            distances = [abs(z_final - root) for root in roots]
            region[i, j] = np.argmin(distances)

    return region, X, Y

def plot_convergence(center, width, resolution, bias=1e-5):
    """
    画图
    """
    region, X, Y = get_convergence_region(center, width, resolution, bias)
    plt.figure(figsize=(7, 6))
    plt.imshow(region, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='Paired')
    plt.colorbar(label='Root index (0: 1, 1: $e^{i\\frac{2\\pi}{3}}$, 2: $e^{i\\frac{4\\pi}{3}}$)')
    center_str = f"({center.real:.2f}{center.imag:+.2f}j)"
    
    plt.title(f"Newton's Method Convergence (Center: {center_str}, Width: {width}, res: {resolution})", fontsize=20)
      
    plt.xlabel('Re(z)',fontsize=20)
    plt.ylabel('Im(z)',fontsize=20)
    plt.show()


# plot_convergence(center=-0.8+0j,width=0.25,resolution=0.0005)
plot_convergence(center=-0.56+0.18j,width=0.1,resolution=0.0002)