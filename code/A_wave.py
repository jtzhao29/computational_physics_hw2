import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("./data/waveform.dat")
time = data[:,0]
voltage = data[:,1]

sampling_rate = 1 / (time[1] - time[0])
n = len(voltage)
freq = np.fft.fftfreq(n, d=1/sampling_rate)
fft_values = np.fft.fft(voltage)

plt.figure(figsize=(12, 6))


plt.plot(freq[:n//2], np.abs(fft_values)[:n//2])
plt.title('Frequency Domain Analysis using FFT',fontsize=25)
plt.xlabel('Frequency (Hz)',fontsize=25)
plt.ylabel('Amplitude',fontsize = 25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.show()

print("the first 4 frequencies:")
print(freq[:4])

def find_the_largest4(Amplitude:np.ndarray)->np.ndarray:
    """\n"""
    Amplitude = np.abs(Amplitude)
    Amplitude = Amplitude[:n//2]
    Amplitude = Amplitude[np.argsort(Amplitude)[::-1]]
    return Amplitude[:4]

print("\namplitude of the largest 4 frequencies:")
print(find_the_largest4(fft_values))

plt.plot(time[:5000], voltage[:5000])
plt.title('Wave',fontsize=25)
plt.xlabel('Time (s)',fontsize=25)
plt.ylabel('Voltage (V)',fontsize = 25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.show()