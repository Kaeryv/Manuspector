import sys
sys.path.append(".")
from dataset import load_spectral_data, dataset_dict_to_dense, augment_mix
import numpy as np
import matplotlib.pyplot as plt

sources = [
    "./raw_data/DataJulie/historique/013-1C-XXX-JB002-X00*.asc",
]
dataset_dict = load_spectral_data(sources, verbose=True)

# We label the dataset and convert it to numpy arrays
X, Y = dataset_dict_to_dense(dataset_dict)

print(X.shape)


fig, ax = plt.subplots()
ax.plot(X.T)
fig.savefig("Figures/spectra.png")