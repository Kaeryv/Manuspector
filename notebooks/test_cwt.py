import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 1 * t) #+ signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 400)
cwtmatr = signal.cwt(sig, signal.ricker, widths)

cwtmatr_yflip = np.flipud(cwtmatr)
plt.imshow(cwtmatr_yflip, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
# plt.yscale("log")
plt.show()
plt.plot(cwtmatr[0])
plt.plot(cwtmatr[-1])
plt.plot(sig)
plt.show()