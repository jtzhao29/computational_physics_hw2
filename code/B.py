import numpy as np
import matplotlib.pyplot as plt

def f(z):
    """
    方程 f(z) = z^3 - 1
    """
    return z**3 - 1

def f_prime(z):
    """
    方程 f'(z) = 3z^2
    """
    return 3 * z**2

def newton_method(z0, max_iter=100, tol=1e-6):
    """
    牛顿迭代法
    z0: 初始猜测值
    max_iter: 最大迭代次数
    tol: 收敛容忍度
    """
    z = z0
    for _ in range(max_iter):
        dz = f(z) / f_prime(z)
        z = z - dz
        if abs(dz) < tol:
            break
    return z

def get_convergence_region(center, width, resolution, bias=1e-5):
    """
    获取在复平面上牛顿法的收敛区域
    center: 中心点 (复数)
    width: 半宽度，表示正方形区域的范围
    resolution: 网格的分辨率
    bias: 收敛的容忍度
    """
    x_min, x_max = center.real - width, center.real + width
    y_min, y_max = center.imag - width, center.imag + width
    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    
    # 创建网格
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = X + 1j * Y
    
    # 根的集合
    roots = [1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)]  # 3个根
    
    # 收敛区域
    region = np.zeros(Z.shape, dtype=int)
    
    # 对每个点应用牛顿法
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            z = Z[i, j]
            # 对每个初始点进行牛顿迭代
            z_final = newton_method(z, max_iter=50, tol=bias)
            # 找到收敛的根
            distances = [abs(z_final - root) for root in roots]
            region[i, j] = np.argmin(distances)  # 收敛到哪个根
    
    return region, X, Y

def plot_convergence(center, width, resolution, bias=1e-5):
    """
    可视化牛顿法收敛区域
    """
    region, X, Y = get_convergence_region(center, width, resolution, bias)
    x1 = np.linspace(center,center+width,1000)
    y1 = x1*(3**0.5)/3
    x0 = np.linspace(center,center,1000)
    x2 = np.linspace(center-width,center,1000)
    y0 = np.linspace(center,center+width,1000)
    y2 = x2*(3**0.5)/3
    y22 = -x2*(3**0.5)/3

    # 绘制收敛图
    plt.figure(figsize=(8, 6))
    plt.imshow(region, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='Paired')
    plt.plot(x1,y1,color = 'black')
    plt.plot(x0,y0,color = 'black')
    plt.plot(x2,y22,color = 'black')
    plt.plot(x2,y2,color = 'black')
    plt.plot(x0,-y0,color = 'black')
    plt.plot(x1,-y1,color = 'black')
    plt.colorbar(label='Root index (0: 1, 1: $e^{i\\frac{2\\pi}{3}}$, 2: $e^{i\\frac{4\\pi}{3}}$)')
    center_str = f"({center.real:.2f}{center.imag:+.2f}j)"
    
    plt.title(f"Newton's Method Convergence (Center: {center_str}, Width: {width}, res: {resolution})", fontsize=20)
      
    plt.xlabel('Re(z)',fontsize=20)
    plt.ylabel('Im(z)',fontsize=20)
    plt.show()


# 测试可视化
plot_convergence(center=0.0+0.0j, width=0.8, resolution=500, bias=1e-5)

# plot_convergence(center=-0.8+0j,width=0.25,resolution=1000)
# plot_convergence(center=-0.56+0.18j,width=0.1,resolution=1000)