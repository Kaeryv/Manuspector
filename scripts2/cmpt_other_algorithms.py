import sys
sys.path.append(".")
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-method", type=str, required=True)
parser.add_argument("-savgol", nargs=2, type=int, default=[11, 2]) # Savgol kernel size and order.
parser.add_argument("-validation", default="historical", type=str)
parser.add_argument("-max_samples", default=5, type=int)
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


Xt, Yt, Xv, Yv = load(args.savgol, args.validation, args.max_samples, args.folds)

def consencus_accuracy(models, X, Y):
    # Concensus
    predictions = []
    for i, model in enumerate(models):
        preds = model.predict(X)
        predictions.append(preds)
    
    predictions=np.asarray(predictions)
    c_preds = list()
    for s in range(len(predictions.T)):
        pred, counts = np.unique(predictions[:, s], return_counts=True)
        pred = pred[np.argmax(counts)]
        c_preds.append(pred)

    c_preds= np.asarray(c_preds)
    acc = np.count_nonzero(c_preds == Y) / len(c_preds)
    return acc


def accuracy_svm():
    from sklearn import svm
    clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(Xt, labels_t.flatten())
    preds = clf.predict(Xv)
    acc = np.count_nonzero(preds == labels_v[:,0]) / len(preds)
    return acc

def accuracy_rf():
    from sklearn.ensemble import RandomForestClassifier
    models = list()
    accuracies = []
    depths = [5, 10]
    for i in range(40):
        model = RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1, random_state=np.random.randint(9999999))
        model.fit(Xt, labels_t.flatten())
        preds = model.predict(Xv)
        acc = np.count_nonzero(preds == labels_v[:,0]) / len(preds)
        accuracies.append(acc)
        models.append(model)
        print("PARAMETERS:", sum(tree.tree_.node_count for tree in model.estimators_) * 5)
    return accuracies

def accuracy_pcaknn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    accuracies = list()
    for i in range(2, 15):#15
        pca = PCA(i)
        Xtp = scaler.fit_transform(Xt)
        Xvp = scaler.transform(Xv)
        Xtp = pca.fit_transform(Xtp)
        Xvp = pca.fit_transform(Xvp)
        models = list()
        model = KNeighborsClassifier(4)
        model.fit(Xtp, labels_t.ravel())
        preds = model.predict(Xvp)
        acc = np.count_nonzero(preds == labels_v[:,0]) / len(preds)
        accuracies.append(acc)
        models.append(model)

    return accuracies

def accuracy_knn():
    from sklearn.neighbors import KNeighborsClassifier
    accuracies = list()
    models = list()
    for i in range(9, 28):#15
        model = KNeighborsClassifier(i)
        model.fit(Xt, labels_t.ravel())
        preds = model.predict(Xv)
        acc = np.count_nonzero(preds == labels_v[:,0]) / len(preds)
        accuracies.append(acc)
        models.append(model)

    return accuracies

if args.method == "KNN":
    accs = accuracy_knn()
elif args.method == "PCAKNN":
    accs = accuracy_pcaknn()
elif args.method == "RF":
    accs = accuracy_rf()
elif args.method == "SVM":
    accs = accuracy_svm()

np.savetxt(f"ext/{args.validation}_{args.method}.acc.txt", accs)

