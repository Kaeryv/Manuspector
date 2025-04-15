from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-method", type=str, required=True)
parser.add_argument("-savgol", nargs=2, type=int, default=[11, 2]) # Savgol kernel size and order.
parser.add_argument("-validation", default="historical", type=str)
parser.add_argument("-max_samples", default=3, type=int)
parser.add_argument("-folds", type=int, default=[4, 0], nargs=2)
parser.add_argument("-seed", type=int, default=42)
args = parser.parse_args()


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from glob import glob
from tqdm import tqdm

from misc import data_augment, count_parameters, torchify
from dataset import load_spectral_data, dataset_dict_to_dense

# Deterministic numpy random behavior (important for k-fold sampling)
np.random.seed(args.seed*543246)

dataset_dict = load_spectral_data([
    "./raw_data/DataDavid/UVVIS/*.asc",
    "./raw_data/DataHenry23/Perkin/New_*/*.asc",
    "./raw_data/DataJulie/07_09_spectro/**/*.asc",
    "./raw_data/DataJulie/14_09_2023/*.asc",
    "./raw_data/DataJulie/11_09_2023/*.asc",
    "./raw_data/DataJulie/14_09_2023_histo/*.asc",
    "./raw_data/DataJulie/historique/*.asc",
], verbose=True)

# We label the dataset and convert it to numpy arrays
X, Y = dataset_dict_to_dense(dataset_dict)

# Build the wavelength vector
wavelength = np.flipud(np.linspace(250, 2500, 226))
# Wavelengths above 2300nm are removed because of low SNR
wl_mask = wavelength <= 2300
X = X[:, wl_mask]
wavelength = wavelength[wl_mask]


if args.savgol[0] > 0:
    X = savgol_filter(X, args.savgol[0], args.savgol[1], axis=-1)

# StandardScaling
dsmean = X.mean(axis=0)
dsstd = X.std(axis=0) * 2
X = (X - dsmean) / dsstd

# Identify every single historic parchment (samples sets 12,13).
histo_mask = (Y[:, 0] == 13) | (Y[:, 0] == 12)

if args.validation == "historical":
    # Train/Test split based on the histo_mask
    Xv = X[histo_mask]
    Yv = Y[histo_mask]
    Xt = X[~histo_mask]
    Yt = Y[~histo_mask]

elif args.validation == "historical+":
    # Train/Test split with histo also in training set but later used only for training the AE
    Xv = X[histo_mask]
    Yv = Y[histo_mask]
    Xt = X.copy()
    Yt = Y.copy()

elif args.validation == "segment":
    # Train/Test split based on k-fold procedure
    validation_ids = np.arange(len(Y))
    np.random.shuffle(validation_ids)
    validation_ids = np.array_split(validation_ids, args.folds[0])
    validation_mask = np.zeros(len(Y)).astype(bool)
    validation_mask[validation_ids[args.folds[1]]] = True
    Xv = X[validation_mask]
    Yv = Y[validation_mask]
    Xt = X[~validation_mask]
    Yt = Y[~validation_mask]

del X, Y

# Limit samples per parchment
Xt = Xt[Yt[:,5]< args.max_samples,:]
Yt = Yt[Yt[:,5]< args.max_samples,:]

Xv = Xv[Yv[:,5]< 5,:]
Yv = Yv[Yv[:,5]< 5,:]

labels_t = Yt[:, 1:2]
labels_v = Yv[:, 1:2]

print("Samples per class (training): ", [np.count_nonzero(labels_t == i) for i in range(3)])
print("Samples per class (validation): ", [np.count_nonzero(labels_v == i) for i in range(3)])


if args.method == "KNN":
    from sklearn.neighbors import KNeighborsClassifier
    knn_accuracies = list()
    for i in range(2, 20):#15
        model = KNeighborsClassifier(i)
        model.fit(Xt, labels_t)
        preds = model.predict(Xv)
        acc = np.count_nonzero(preds == labels_v[:,0]) / len(preds)
        knn_accuracies.append(acc)

    print([round(acc,4) for acc in knn_accuracies ])
