import sys
sys.path.append(".")
import numpy as np
from dataset import load_spectral_data, dataset_dict_to_dense
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

from matplotlib import font_manager
import matplotlib as mpl

import matplotlib.pyplot as plt
from pathlib import Path
font_path = Path("times.ttf")
#plt.rcParams["font"] = font


import matplotlib.font_manager as font_manager
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
#font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
#font_list = font_manager.createFontList(font_files)
#font_manager.fontManager.ttflist.extend(font_list)

#mpl.rcParams['font.family'] = 'times'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()




from scipy.signal import ricker, cwt, convolve
def reconstruct(C, widths):
    return np.sum([ convolve(C[i], ricker(C.shape[1], w), mode="same")/w**2 for i, w in enumerate(widths) ], axis=0)

def filtered(sig, offset, octaves):
    widths = offset + 1 + np.arange(octaves)
    C = cwt(sig, ricker, widths)
    return reconstruct(C, widths)



# We label the dataset and convert it to numpy arrays

if not os.path.isfile("xtmp.npz"):
    validation_indices = list()
    # Training data
    with open("data.txt") as f:
        sources = f.read().splitlines()
        # Find historical samples
        for s in sources:
            if "VALIDATION" in s:
                validation_indices = list(map(int, s.split()[1:]))
        sources = [ s for s in sources if ~s.startswith("#")]
        dataset_dict = load_spectral_data(sources, verbose=True)
    
    names = list(dataset_dict.keys())
    X, Y = dataset_dict_to_dense(dataset_dict)
    
    # Build the wavelength vector
    wavelength = np.flipud(np.linspace(250, 2500, 226))
    # Wavelengths above 2300nm are removed because of low SNR
    wl_mask = wavelength <= 2300
    X = X[:, wl_mask]
    wavelength = wavelength[wl_mask]
    
    octaves = 128
    offset = 0
    
    def do_cwt(data, octaves=128, offset=1, stridesf=1, stridesx=1):
        cwts = list()
        for i in range(len(data)):
            cwts.append(cwt(data[i], wavelet=ricker, widths=np.arange(offset+1, offset+octaves+1)))
        cwts = np.asarray(cwts) / 5
        return np.abs(cwts[:, ::stridesf, ::stridesx]).real #[:, offset:offset+octaves]
    X1D = X.copy()
    X = do_cwt(X, octaves=octaves, offset=offset, stridesf=1, stridesx=4)
    
    # StandardScaling
    dsmean = X.mean(axis=(0,1))
    dsstd = X.std(axis=(0,1)) * 2
    X = (X - dsmean) / dsstd
    np.savez_compressed("xtmp.npz", X=X, dsstd=dsstd, dsmean=dsmean, X1D=X1D, octaves=octaves, wavelength=wavelength, Y=Y)
else:
    data = np.load("xtmp.npz")
    X = data["X"]
    X1D = data["X1D"]
    wavelength = data["wavelength"]

sample_spectrum = 2
perf_scalogram_offset = np.load("opt.npz")["c"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 3.5), dpi=250, sharex=True, constrained_layout=True)
vmax = abs(X[sample_spectrum]).max()
scales = np.linspace(0.010, 1.28,128)
im1 = ax1.matshow(X[sample_spectrum].T, 
        cmap="RdBu", 
        extent=[0.01, 1.28, wavelength[-1]/1000, wavelength[0]/1000] , 
        origin="lower", vmax=vmax, vmin=-vmax, 
        aspect=0.21)
ax1.text(-0.1, 1.05, "(A)", transform=ax1.transAxes, size=16, weight='normal')
ax1.set_xlabel("Wavelet scale [µm]")
ax1.set_ylabel("Wavelength [µm]")

cb = fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.03, shrink=0.8,pad=-0.35)
cb.ax.set_ylabel("Normalized CWT \nmagnitude")
labels = ["240nm", "320nm", "480nm", "640nm"]
for profile, label in zip(perf_scalogram_offset, labels):
    ax2.plot(scales, profile,label=label)
ax2.set_xlabel("Scalogram window position [µm]")
ax2.set_ylabel("Historical holdout accuracy", fontsize=8)
ax2.text(-0.1, 1.02, "(B)", transform=ax2.transAxes, size=16, weight='normal')

ax2.legend(title="Window width $\sigma$",loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig("Figures/fig-wavelet-opt.png")
plt.savefig("Figures/fig-wavelet-opt.pdf")
plt.savefig("Figures/fig-wavelet-opt.svg")
