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
data = None
for i in range(1,10):
    db = np.load(f"shapley_data/shap_values_{i}.npz")
    #maxacc = np.median(torch.load(models_folder+f"/{10}_{i}.pkl")["accs"])
    #if maxacc > 0.:
    shaps.append(db["values"])
    labs.append(db["labels"])
    if data is None:
        data = db["data"]

labs = np.asarray(labs)
lab = labs[0]
shaps = np.asarray(shaps)
shap = shaps.mean(0)
names = ["Calf", "Sheep", "Goat" ]
import scipy.ndimage as ndimage

def cm(length):
    return length / 2.54
def get_species_shap(mean, index, which="lower"):
    mask_species = lab==index
    shap_species = shap[index, mask_species, 0]
    # 50, 1, 16, 25) (1, 16, 25) (3, 17, 16, 25)
    if which == "lower":
        mask = data[mask_species,0] < np.repeat(mean, shap_species.shape[0], axis=0)
    else:
        mask = data[mask_species,0] >= np.repeat(mean, shap_species.shape[0], axis=0)
    shap_species[~mask] = 0
    #print("L", i, data.shape, mean.shape, shap_species.shape, mask.shape)

    shap_species = np.nanmean(shap_species, axis=0) # Mean over samples that are species i
    shap_species = ndimage.gaussian_filter(shap_species[:,:,np.newaxis], sigma=(3, 3, 0), order=0, )[:,:,0]

    return shap_species
   
which = ["lower", "higher"]
which_label = ["$R < \\bar R$", "$R >= \\bar R$"]
from mpl_toolkits.axes_grid1 import make_axes_locatable


fig, axs = plt.subplots(3, 3, figsize=(cm(18),cm(11)), sharex=True, dpi=200)
for j in range(2):
    for i, ax in enumerate(axs[j+1,:3]):
        mean = data.mean(axis=0)
        shap_species = get_species_shap(mean, i, which=which[j])
        shap_species_other = get_species_shap(mean, i, which=which[j-1])
        importance = np.sum(np.abs(shap_species) + np.abs(shap_species_other), axis=0)
        vmax = np.max(np.abs(shap_species))*0.8
        #im = ax.contourf(shap_species, cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower", extent=[np.max(wl), np.min(wl), 0.1, 0.64])
        ax.bar(wl[::2], np.sum(shap_species, axis=0), width=0.02, alpha=0.2, color="k")
        #im = ax.matshow(shap_species, cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower", extent=[np.max(wl), np.min(wl), 10, 64])
        #ax.set_aspect(2)
        if i == 1:
            ax.set_title(which_label[j])

for i, ax in enumerate(axs[0,:]):
    ax.set_title(names[i])
#divider = make_axes_locatable(axs[2, 2])
#cax = divider.append_axes('right', size='5%', pad=0.05)
ax = fig.add_axes((0.9,0.1,0.01,0.5))
#fig.colorbar(im, cax=ax, orientation='vertical')
#ax.set_yticks([-0.0006, 0.0, 0.0006])
ax.set_yticklabels([f"{round(xx*100000,1)}E-5" for xx in ax.get_yticks()])
from matplotlib.ticker import LogFormatter as LF
from matplotlib.ticker import MultipleLocator as ML
from matplotlib.ticker import ScalarFormatter as SF
#ax.yaxis.set_minor_formatter(LF())


pics = [
    AnnotationBbox(OffsetImage(plt.imread("./cow.png", format='png'), zoom=0.15), [500, 25], xybox=(1800, 80), xycoords='data', frameon=False,),
    AnnotationBbox(OffsetImage(plt.imread("./sheep.png", format='png'), zoom=0.15), [500, 25], xybox=(1800, 80), xycoords='data', frameon=False,),
    AnnotationBbox(OffsetImage(plt.imread("./goat.png", format='png'), zoom=0.15), [500, 25], xybox=(1800, 80), xycoords='data', frameon=False,),
]
colors = [ "#567ab4", '#d6bd73', "#c0736d"]
for i, ax in enumerate(axs[0,:3]):
    ax.add_artist(pics[i])
    #ax.plot(wl, specmean[wl_mask], ls=":", c="k")
    for j in range(3):
        ax.plot(wl, samples_spectra[j], alpha=0.2 if i != j else 1.0, color=colors[j])

for ax in axs[:3,1:].flat:
    ax.set_yticklabels([])
for ax in axs[2,0:].flat:
    ax.set_xlabel("Wavelength [µm]")

for ax in axs.flat:
    ax.set_xlim(np.min(wl), np.max(wl))
    #ax.xaxis.set_minor_locator(ML(400))
    ax.xaxis.set_major_locator(ML(0.4))
    #ax.xaxis.set_minor_formatter(SF())
    #ax.tick_params(axis='x', which="minor", rotation=90, labelsize=8)
    ax.tick_params(axis='x', which="major", rotation=45, labelsize=8)
    
#axs[1,2].contourf(data[0,0], cmap="RdBu", origin="lower", extent=[np.max(wl), np.min(wl), 10, 64])
axs[1,0].set_ylabel("Wavelet scale [µm]")
axs[2,0].set_ylabel("Wavelet scale [µm]")
axs[0,0].set_ylabel("Reflectance [%]")
plt.subplots_adjust(left=0.1, right=0.88, wspace=0.03, hspace=0.25, top=0.9,bottom=0.08)

#shap_all_species = np.abs(shap)[:, :,0].mean(1)
#shap_all_species = shap_all_species.sum(0)
#shap_all_species = ndimage.gaussian_filter(shap_all_species[:,:,np.newaxis] , sigma=(1, 3, 0), order=0)
##axs[3].contour(shap_all_species[:,:,0], origin="lower", colors=["black"])
#C = cwt(spectrum_sheep, ricker, range(10,10+64))
#axs[0,3].matshow(C, origin="lower", extent=[np.max(wl), np.min(wl), 0, 1])
##axs[3].matshow(data[0,0], origin="lower", extent=[np.max(wl), np.min(wl), 0, 1])
#axs[0,3].set_aspect(400)
#axs[0,4].plot(wl[::2], spectrum_sheep[::2]/100)
#axs[0,4].plot(wl[::2], spectrum_calf[::2]/100)
# Annotate with arrows

import matplotlib.patches as mpatches
x_tail = 0.7
y_tail = 0.5
x_head = 0.7
y_head = 0.7
dx = x_head - x_tail
dy = y_head - y_tail

arrow = mpatches.FancyArrowPatch((0.7, 0.4), (0.7, 0.1),
        mutation_scale=25, transform=axs[0,0].transAxes)
axs[0,0].add_patch(arrow)
arrow = mpatches.FancyArrowPatch((0.7, 0.1), (0.7, 0.4), 
        mutation_scale=25, transform=axs[0,1].transAxes)
axs[0,1].add_patch(arrow)
arrow = mpatches.FancyArrowPatch((0.2, 0.4), (0.2, 0.1), 
        mutation_scale=25, transform=axs[0,1].transAxes)
axs[0,1].add_patch(arrow)

arrow = mpatches.FancyArrowPatch((0.6, 0.4), (0.6, 0.1), 
        mutation_scale=25, transform=axs[0,2].transAxes)
axs[0,2].add_patch(arrow)
arrow = mpatches.FancyArrowPatch((0.3, 0.1), (0.3, 0.4), 
        mutation_scale=25, transform=axs[0,2].transAxes)
axs[0,2].add_patch(arrow)


plt.savefig("Figures/shap2.png")
plt.savefig("Figures/shap2.pdf")

'''
017-4C-XXX-NR005-C001
017-4C-XXX-NR005-F001
017-2G-XXX-NR005-F001
017-2G-XXX-NR005-C001
017-2S-XXX-NR005-F001
017-2C-XXX-NR005-C001
017-2C-XXX-NR005-F001

# Black ref
017-2G-XXX-NR006-C001
017-2G-XXX-NR006-F001
017-2C-XXX-NR006-F001
017-2C-XXX-NR006-C001
017-4C-XXX-NR006-C001
017-4C-XXX-NR006-F001
017-2S-XXX-NR006-F001
017-2S-XXX-NR006-C001

'''

#fig, (ax1, ax2) = plt.subplots(2)
#ax1.plot(X1D[0])
#ax2.matshow(X[0], origin="lower")
#plt.savefig("Figures/shap2.png")
