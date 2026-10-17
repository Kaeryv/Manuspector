import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from dataset import read_asc
from scipy.signal import ricker, cwt, convolve
from matplotlib.offsetbox import (OffsetImage,AnnotationBbox)
import torch

samples_spectra = [
    "raw_data/DataNicolas/17_01_2024/017-4C-XXX-NR006-F001.Sample.asc",
    #"raw_data/DataHenry23/Perkin/New_Parchment/002-3C-H10-HP001-F001.Sample.Raw.asc",
    "raw_data/DataHenry23/Perkin/New_Parchment/001-3S-N10-HP001-F001.Sample.Raw.asc",
    "raw_data/DataHenry23/Perkin/New_Parchment/003-3G-N10-HP001-F002.Sample.Raw.asc",
    ]

models_folder="output/offset_sweep_sgd_w64r05_smp_bis/"
wl = np.flipud(np.linspace(250, 2500, 226))
wl_mask = wl <= 2300
wl = wl[wl_mask] / 1000
samples_spectra = [ read_asc(f)[1][wl_mask] for f in samples_spectra ]
shaps = []
labs = []
datas = []
backgrounds = []
num_shap_instances = 9

for i in range(1,num_shap_instances+1):
    db = np.load(f"shapley_data/shap_values_{i}.npz")
    shaps.append(db["values"])
    labs.append(db["labels"])
    datas.append(db["data"])
    backgrounds.append(db["background"])

labs = np.asarray(labs).reshape(9*100, -1)
shaps = np.asarray(shaps)
datas = np.asarray(datas)
backgrounds = np.asarray(backgrounds)
#shap = shaps.mean(0)
names = ["Calf", "Sheep", "Goat" ]
import scipy.ndimage as ndimage


data = np.load("xtmp.npz")
X1D = data["X1D"]
Y = data["Y"]
wavelength = data["wavelength"]
sample_limiter = Y[:,5]< 5
X1D = X1D[sample_limiter,:]
Y = Y[sample_limiter,:]



def cm(length):
    return length / 2.54
   
which = ["lower", "higher"]
which_label = ["$R < \\bar R$", "$R >= \\bar R$"]
from mpl_toolkits.axes_grid1 import make_axes_locatable
shaps = np.swapaxes(shaps, 0, 1)
shaps = shaps.reshape(3, 9*100, 1, 64, 103)
from scipy.signal import savgol_filter
datas = datas.reshape(9*100, 1, 64, 103)

fig, axs = plt.subplots(2, 3, figsize=(cm(19),cm(11)), sharex=True, dpi=200)
backgrounds = backgrounds.reshape(9*100,*backgrounds.shape[2:])
from scipy.stats import pearsonr
for i, ax in enumerate(axs[1,:3]):
    # sample = labs[:,0]
    # histo = (sample == 12)|(sample==13)|(sample==15)|(sample==16)|(sample==19)
    species_mask = (labs[:, 1] == i) #& ~histo
    nsamples = np.count_nonzero(species_mask)
    species_shap_sg = shaps[i, species_mask] # SHAP for species i for samlpes of said species
    species_spectra_sg = backgrounds[species_mask]

    species_shap = np.sum(species_shap_sg, axis=(1,2))
    species_spectra = np.sum(species_spectra_sg, axis=(1,2))

    wavelength = wl[::2]
    wavelengths = np.tile(wavelength, nsamples).reshape(nsamples, -1).flatten()
     
    C = [ pearsonr(species_shap[:,j].flatten(), species_spectra[:,j].flatten())[0] for j in range(len(wavelength))]
    I = np.sum(np.abs(species_shap_sg), axis=(0,1)).mean(0)

    vmax = np.max(species_spectra)
    ss = np.abs(species_spectra.flatten()) > 0
    print("Filtered", np.count_nonzero(ss), len(ss), " of the points.")
    print(np.mean(species_spectra))
    im = ax.scatter(wavelengths[ss], 10*species_shap.flatten()[ss], 
        cmap="RdBu", c=species_spectra.flatten()[ss], alpha=0.08,s=10,marker='.', vmin=-vmax,vmax=vmax, rasterized=True)

    C = savgol_filter(C, 7, 2)
    C_col = "#434C5E"
    I_col = "#A3BE8C"
    p1, = ax.plot(wavelength, C, color=C_col)
    ax.set_ylim(-0.75, 0.75)

    ax2 = ax.twinx()
    if i < 2:
        ax2.set_yticks([])
        if i == 0:
            ax2.spines['left'].set_color(C_col)
            ax.tick_params(axis='y', colors=C_col)
    else:
        ax2.spines['right'].set_color(I_col)
        ax2.tick_params(axis='y', colors=I_col)

    
    I = savgol_filter(I, 7, 2)
    ax2.set_ylim(0.0, 0.5)
    p2, = ax2.plot(wavelength, I, color=I_col)

plt.legend([p1,p2], ["Pearson correlation","SHAP Importance"],ncols=2, bbox_to_anchor=(-1.5, 1.25), loc="upper left")

for i, ax in enumerate(axs[0,:]):
    ax.set_title(names[i])

ax.set_yticklabels([f"{round(xx,1)}" for xx in ax.get_yticks()])

from matplotlib.ticker import LogFormatter as LF
from matplotlib.ticker import MultipleLocator as ML
from matplotlib.ticker import ScalarFormatter as SF


colors = [ "#567ab4", '#d6bd73', "#c0736d"]
specmean = X1D.mean(0)
for i, ax in enumerate(axs[0,:3]):
    ax.plot(wl, specmean, ls=":", c="k")
    for j in range(3):
        species_mask = (Y[:,1] == j)
        mean = X1D[species_mask].mean(0)
        ax.plot(wl, mean, alpha=0.2 if i != j else 1.0, color=colors[j])

for ax in axs[:3,1:].flat:
    ax.set_yticklabels([])
for ax in axs[1,0:].flat:
    ax.set_xlabel("Wavelength [µm]")

for ax in axs.flat:
    ax.set_xlim(np.min(wl), np.max(wl))
    ax.xaxis.set_major_locator(ML(0.4))
    ax.tick_params(axis='x', which="major", rotation=45, labelsize=8)
    
axs[1,0].set_ylabel("Correlation coefficient")
ax2.set_ylabel("Absolute SHAP Value")
ax2.yaxis.label.set_color(I_col)
axs[1,0].yaxis.label.set_color(C_col)

axs[0,0].set_ylabel("Reflectance [%]")
plt.subplots_adjust(left=0.1, right=0.88, wspace=0.03, hspace=0.25, top=0.9,bottom=0.12)

plt.savefig("Figures/shap4.png")
plt.savefig("Figures/shap4.pdf")
