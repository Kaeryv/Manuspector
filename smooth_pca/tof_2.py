from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-folder", type=str)
parser.add_argument("-action", type=str, required=True)
parser.add_argument("-name", type=str)
parser.add_argument("-species", type=str, nargs="+", default=["S", "C", "G"])
args = parser.parse_args()


import numpy as np
only = [ "3CG14-F", "3SG13-F","3GG13-F" ] # "3CG14-G", "3SG13-G", "3GG13-G", 
labels2int = { "S": 0, "C": 1, "G": 2 }
colorsi = ["#DBBD86", "#4F81BD", "#CC6D68"]
gf2int = { "G": 0, "F": 1 }
if args.action == "ds":
    from glob import glob
    from os.path import basename
    data = list()
    files = glob(args.folder + "/*")
    from tqdm import tqdm
    tokens = list()
    ds = list()
    labels = list()
    labels_gf = list()
    for i, (file) in enumerate(tqdm(files)):
        instance  = dict()
        token = basename(file).split("_")[0]
        instance["name"] = token
        instance["species"] = token[1]
        instance["spot"] = token[2:5]
        instance["flesh"] = gf2int[token[6]] == 1
        spectre = np.loadtxt(file, skiprows=3)[200311:826893:1, 2]
        masses = np.loadtxt(file, skiprows=3)[200311:826893:1, 1]
        instance["spectre"] = spectre.tolist()
        instance["masses"] = masses.tolist()
        data.append(instance)
    import json
    with open(args.name, "w") as f:
        json.dump(data, f)

elif args.action == "pca":
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    ds = np.load("ds.npz")
    x = ds["ds"]
    x[x < 200.]= 0.0
    x /= np.max(x)
    scaler = StandardScaler()
    model = PCA(n_components=len(x), svd_solver='randomized', random_state=2)
    x = scaler.fit_transform(x)
    x = model.fit_transform(x)
    np.savez_compressed(args.name + ".npz", x=x, evr=model.explained_variance_ratio_, labels=ds["labels"], labels_gf=ds["labels_gf"])

elif args.action == "tsne":
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    ds = np.load("ds.npz")
    x = ds["ds"]
    scaler = StandardScaler()
    model = TSNE(n_components=2)
    x = scaler.fit_transform(x)
    x = model.fit_transform(x)
    np.savez_compressed(args.name+".npz", x=x, evr=[], labels=ds["labels"], labels_gf=ds["labels_gf"])

elif args.action == "graph":
    import matplotlib.pyplot as plt
    d = np.load(args.name + ".npz")
    x = d["x"]
    evr = d["evr"]
    labels = d["labels"]
    labels_gf = d["labels_gf"]
    print(x.shape)
    for i in range(3):
        plt.scatter(*x[labels==i, 0:2].T, c=colorsi[i])
    plt.savefig(args.name + ".png")
    if len(evr) > 0:
        plt.figure()
        plt.plot(np.arange(len(evr)), evr, "r")
        plt.savefig("skree2.png")
    print(x.shape)

# Identifier variance intraclass sur chaque espèce
# - via loadings PCA
# - algo de classif + shap
