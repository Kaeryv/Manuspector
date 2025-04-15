from glob import glob
import numpy as np
from tqdm import tqdm
from os.path import basename, splitext
root = "./data/"
#root = "C:/Users/...."

files = glob(f"{root}/**/*.[Aa][Ss][Cc]", recursive=True)
X = list()
L = list()
print("Processing", len(files), " files")
#exit()
for i, f in enumerate(tqdm(files)):
    name = splitext(basename(f))[0]
    if "Peau" in name or "Vierge" in name:
        continue
    pid = int(name.split("-")[0][1:])
    cid = 0
    if "Rouge" in name:
        cid = 0
    elif "Vert" in name:
        cid = 1
    elif "Bleu" in name:
        cid = 2
    elif "Noir" in name:
        cid = 3
    elif "Or" in name:
        cid = 4
    else:
        assert(False, f"Don't know the color: {name}")
    L.append((pid,cid))
    #print(name)
    # :, 0 -> wvl
    # :, 1 -> absorbance
    #print(i, f, data.shape)

    data = np.loadtxt(f, skiprows=25) # ! Perkin
    X.append(data)

# Tableau contigu
X = np.asarray(X)
L = np.asarray(L)

np.savez("db.npz", X=X, L=L)



