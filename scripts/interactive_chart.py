import sys
sys.path.append(".")
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--data", type=str, default="./data/parchments.npz")
parser.add_argument("--scaler", type=str, required=True, default="default")
parser.add_argument("--classif", type=str)
parser.add_argument("--subsample", type=int, default=2)
args = parser.parse_args()

import time
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Encoder, self).__init__()
        self.l1 = nn.Linear(in_features, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, out_features)
    def forward(self, X):
        X = torch.tanh(self.l1(X))
        X = torch.relu(self.l2(X))
        X = torch.tanh(self.l3(X))
        return X

class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Decoder, self).__init__()
        self.l1 = nn.Linear(in_features, 64)
        self.l2 = nn.Linear(64, 256)
        self.l3 = nn.Linear(256, out_features)

    def forward(self, X):
        X = torch.tanh(self.l1(X))
        X = torch.relu(self.l2(X))
        X = torch.tanh(self.l3(X))
        return X

class AutoEncoder(nn.Module):
    def __init__(self, features, latent_features=2, variationnal=False, smoothing=15):
        super(AutoEncoder, self).__init__()
        if variationnal:
            self.enc = Encoder(features, latent_features*2)
            self.dec = Decoder(latent_features, features+smoothing-1)
        else:
            self.enc = Encoder(features, latent_features)
            self.dec = Decoder(latent_features, features+smoothing-1)

        self.latent_features = latent_features
        self.variationnal = variationnal
        self.smoothing = smoothing
        self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.smoothing)
    def forward(self, X):
        bs = X.shape[0]
        if self.variationnal:
            X = self.enc(X).view(-1, 2, self.latent_features)
            mu = X[:, 0, :]
            logvar = (X[:, 1, :] - 1) * 5
            X = self.reparameterize(mu, logvar)
            return self.c1(self.dec(X).view(bs, 1, -1)).view(bs, -1), mu, logvar
        else:
            return self.c1(self.dec(self.enc(X)).view(bs, 1, -1)).view(bs, -1)
    
    def reconstruct(self, X):
        bs = X.shape[0]
        recons = self.dec(X).view(bs, 1, -1)
        recons = self.c1(recons)
        return (recons.view(bs, -1))
    
    def encode(self, X, std=False):
        code = self.enc(X)
        if self.variationnal:
            view = code.view(-1, 2, self.latent_features)
            if std:
                return view[:, 0, :], view[:, 1, :]
            else:
                return view[:, 0, :]
        else:
            return self.enc(X)

    def reparameterize(self, latent_mean, latent_logvar):
        std = torch.exp(0.5*latent_logvar)
        eps = torch.randn_like(std)
        sample = latent_mean + (eps * std)
        return sample
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from toolkit.tools import find_closest_point

# Chargement du modèle mis au point.
data = np.load(args.data)
spectrums = data['x']
labels = data['labels']
from sklearn.preprocessing import MinMaxScaler
scaler = pickle.load(open(args.scaler, "rb"))
x = spectrums[:,:, ::5]
xs = scaler.transform(x.reshape(x.shape[0], -1))
model = AutoEncoder(xs.shape[1], 2, variationnal=True)
model.load_state_dict(torch.load(args.model))
model.eval()
wl = np.flipud(np.linspace(350, 2350, xs.shape[1] // 2))

codes = model.encode(torch.from_numpy(xs).float())
scat_data = codes.detach().numpy()
markers = ["v", "^", "X"]
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,6))
markerlabel = np.array([ a * 10 + b for a, b in zip( labels[:, 2], labels[:, 3])])
for i, c in enumerate(np.unique(markerlabel)):
    mask = markerlabel == c
    cur = scat_data[mask,:]
    ax1.scatter(*cur.T, c=[f"C{l}" for l in labels[mask, 0].flat], marker=markers[i], s=100)
    legend_elements = [Line2D([0], [0], marker="v", markersize=12, color='k', linestyle="", label='Modern Parchments Flesh'),
                   Line2D([0], [0], marker="^", markersize=12, color='k', linestyle="", label='Modern Parchments Grain'),
                   Line2D([0], [0], marker='X', color='k', linestyle="", label='Manuscripts', markersize=12),
                   Patch(facecolor='C0',  label='Agneau'),
                   Patch(facecolor='C1',  label='Veau'),
                   Patch(facecolor='C2',  label='Chèvre'),
                   ]
             

ax1.axis('square')
ax1.set_title(f"Scatter of the samples in latent (coding) space. {0}")
ax1.set_ylabel("Coding dimension 2 of 2.")
ax1.set_xlabel("Coding dimension 1 of 2.")
ax1.legend(handles=legend_elements, bbox_to_anchor =(0.65, 1.15), ncol=2, fancybox=True, title="Legend")

cursor = Cursor(ax1, useblit=True, color='black', linewidth=1, alpha=0.4)



handles = ax2.plot(wl, spectrums[0,0,::5], wl, spectrums[0,1,::5])
handles_closest = ax2.plot(wl, spectrums[0,0,::5], wl, spectrums[0,1,::5])

dot_handle, = ax1.plot(0, 0, 'k', ms=16, marker=4)

ax2.axis([np.min(wl), np.max(wl), 0, 100])
ax2.legend(handles, ('Diffuse reflectance', 'Absorption'))

selected_class = 0
selected_spectrum = 0
def onclick(event):
    global selected_class, selected_spectrum
    if event.xdata is not None:
        reconstructed = model.reconstruct(torch.tensor([[event.xdata, event.ydata]]).float()).detach().numpy()
        ax1.set_title(f"Scatter of the samples in latent (coding) space. {round(event.xdata, 3)}:{round(event.ydata, 3)}")

        reconstructed = scaler.inverse_transform(reconstructed.reshape((1, -1))).reshape(1, 2, -1)
        handles[0].set_ydata(reconstructed[0, 0, :]*100) 
        handles[1].set_ydata(reconstructed[0, 1, :])

        closest = find_closest_point(scat_data, np.array([event.xdata, event.ydata]))
        dot_handle.set_xdata(scat_data[closest, 0])
        dot_handle.set_ydata(scat_data[closest, 1])
        
        if closest != selected_spectrum:
            spectrum = spectrums[closest].reshape((2, -1))[:, ::5]
            handles_closest[0].set_ydata(spectrum[0]*100)
            handles_closest[1].set_ydata(spectrum[1])

            selected_spectrum = closest
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', onclick)
plt.subplots_adjust()
plt.show()


