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

Xt, Yt, Xv, Yv = load(args.savgol, args.validation, 5, args.folds)


from sklearn.neighbors import KNeighborsClassifier
accuracies = list()
k = np.arange(1, 32)
for i in k:
    model = KNeighborsClassifier(i)
    model.fit(Xt, Yt[:,1].ravel())
    preds = model.predict(Xv)
    acc = np.count_nonzero(preds == Yv[:,1]) / len(preds)
    accuracies.append(acc)
best=np.argmax(accuracies)
print("KNN", best, k[best], accuracies[best])
