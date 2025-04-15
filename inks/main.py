import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

data = np.load("db.npz")
X = data["X"][:,:,1] # Seulement la seconde colonne
#xm = X.mean()
#xs = (X-xm).std()
#X = (X - xm) / xs

#scaler = StandardScaler()
#X = scaler.fit_transform(X)
print(X.shape)
L = data["L"]
pca = PCA(n_components=3)
#pca = TSNE(n_components=2, perplexity=20)
x = pca.fit_transform(X)[:,1:]
print(pca.explained_variance_ratio_)
#print(pca.singular_values_)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,1, figsize=(5,5))
axs[0].set_title("Color")
axs[0].scatter(x[:, 0], x[:, 1], c=L[:,1], alpha=0.4)
axs[1].set_title("PID")
axs[1].scatter(x[:, 0], x[:, 1], c=L[:,0], cmap="tab20")
plt.savefig("TSNE.png")
plt.show()
