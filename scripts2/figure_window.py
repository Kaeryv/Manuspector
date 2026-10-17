import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import product
import os

# Adaptive Fourier Neural Operator
def meanfilt (x, k):
    """Apply a length-k mean filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """

    import numpy as np

    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."

    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.mean (y, axis=1)
def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    import numpy as np

    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

N=128
S=10
data = [
    ("output/review_nouv/", False),
    #("output/review_full_64/", False),
    ("output/review_uvvis_64/", False),
    #("output/review_nir_64_2/", False),
    ("output/review_nir_64/", False),
    ("output/final_64/", False),
    #("output/final2_64/", False),
]

def get_acc_evo(root, kfold=False):
    if kfold:
        data = product(range(N), range(4), range(1,S+1))
        max_accuracies = [
            np.nanmax(torch.load(f"{root}/{i}_{j}_{k}.pkl")["accs"]) 
            for (i, j, k) in data 
            ]
        return np.asarray(max_accuracies).reshape(N,4,S).mean(1)
    else:
        data = product(range(N), range(1,S+1))
        max_accuracies = [
            np.nanmax(torch.load(f"{root}/{i}_{j}.pkl")["accs"]) 
            if os.path.isfile(f"{root}/{i}_{j}.pkl") else np.nan
            for (i, j) in data  
            ]
        return np.asarray(max_accuracies).reshape(N,S)

cm = plt.cm.get_cmap('tab10')
curves = []
labels = ["UV removed", "UV and Vis only", "NIR Only", "UV-Vis-NIR"]
for i, (d, iskf) in enumerate(data):
    max_accuracies = get_acc_evo(d, iskf)
    max_accuracies[max_accuracies<0.5] = np.nan
    max_accuracies = np.split(max_accuracies, 2, axis=-1)
    if i == 3:
        max_accuracies = np.nanmax(max_accuracies, axis=0)
    else:
        max_accuracies = np.nanmean(max_accuracies, axis=0)
    curve = meanfilt(np.nanmax(max_accuracies,-1),3)
    plt.plot(np.arange(len(curve))/100, curve, label=labels[i], color=cm.colors[i], marker=".",ls=None)#d
    #plt.plot(np.nanmean(max_accuracies,-1), color=cm.colors[i], ls=":")
    curves.append(curve)
plt.legend()
plt.title("Holdout accuracy")
plt.xlabel("CWT offset [um]")
plt.ylabel("Accuracy [norm]")
plt.yticks([0.6,0.65,0.70,0.75,0.8,0.85])
#plt.axis([0,64,0.5,1])
np.savez("opt.npz", c=curves)
plt.savefig("Figures/cwt_window.png")
