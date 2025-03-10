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