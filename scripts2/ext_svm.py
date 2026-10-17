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

Xt, Yt, Xv, Yv, _, _ = load(args.savgol, args.validation, 5, args.folds)


from sklearn import svm
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(Xt, Yt[:,1].flatten())
preds = clf.predict(Xv)
acc = np.count_nonzero(preds == Yv[:,1]) / len(preds)
print("SVM", acc)
