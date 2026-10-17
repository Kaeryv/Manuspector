import sys
sys.path.append(".")


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-savgol", nargs=2, type=int, default=[11, 2]) # Savgol kernel size and order.
parser.add_argument("-validation", default="historical", type=str)
parser.add_argument("-folds", type=int, default=[4, 0], nargs=2)
parser.add_argument("-seed", type=int, default=42)
args = parser.parse_args()



import numpy as np
from tqdm import tqdm
from dataset import load

np.random.seed(args.seed*543246)

Xt, Yt, Xv, Yv, X, Y = load(args.savgol, args.validation, 5, args.folds)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
accuracies = list()
nd = np.arange(2, 30, 1)
k = np.arange(1, 15)
params = list()
for i in nd:
    pca = PCA(i)
    scaler.fit(X)
    Xtp = scaler.transform(Xt)
    Xvp = scaler.transform(Xv)
    Xtp = pca.fit_transform(Xtp)
    Xvp = pca.fit_transform(Xvp)
    for ki in k:
        model = KNeighborsClassifier(ki)
        model.fit(Xtp, Yt[:,1].ravel())
        preds = model.predict(Xvp)
        acc = np.count_nonzero(preds == Yv[:,1]) / len(preds)
        accuracies.append(acc)
        params.append((i,ki))
accuracies = np.asarray(accuracies)

best=np.argmax(accuracies)
print(best)
print("PCA-KNN", best, params[best], accuracies[best])
