from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-num", type=int, required=True)
args = parser.parse_args()
import numpy as np
from toolkit.dataset import load_spectral_data, dataset_dict_to_dense
import torch

torch.manual_seed(args.num)

def torchify(X, dtype=torch.float):
    return torch.from_numpy(X).type(dtype)


dataset_dict = load_spectral_data([
    "../raw_data/DataDavid/UVVIS/*.asc",
    "../raw_data/DataHenry23/Perkin/New_*/*.asc",
    "../raw_data/DataJulie/07_09_spectro/**/*.asc",
    "../raw_data/DataJulie/14_09_2023/*.asc",
    "../raw_data/DataJulie/11_09_2023/*.asc",
    "../raw_data/DataJulie/14_09_2023_histo/*.asc",
    "../raw_data/DataJulie/historique/*.asc",
], verbose=True)

X, Y = dataset_dict_to_dense(dataset_dict)
valid_mask = np.logical_or(Y[:, 0] == 13, Y[:, 0] == 12)
Xv = X[valid_mask]
Yv = Y[valid_mask]
X = X[~valid_mask]
Y = Y[~valid_mask]

wavelength = np.flipud(np.linspace(250, 2500, 226))
wl_mask = wavelength <= 2300
wavelength = wavelength[wl_mask]

X = X[:, wl_mask]
X = X[Y[:,5]< 5,:]
Y = Y[Y[:,5]< 5,:].astype(np.int32)

print(np.count_nonzero(Y[:, 1] == 0))
print(np.count_nonzero(Y[:, 1] == 1))
print(np.count_nonzero(Y[:, 1] == 2))

Xv = Xv[:, wl_mask]
Xv = Xv[Yv[:,5]< 5,:]
Yv = Yv[Yv[:,5]< 5,:].astype(np.int32)

labels = Y[:, 1]
labels_v = Yv[:, 1]

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch

scaler = StandardScaler()
dsmean = X.mean(axis=0)
dsstd = X.std(axis=0) * 2
Xs = (X - dsmean) / dsstd
#Xs = scaler.fit_transform(X) / 5
#Xsv = scaler.transform(Xv) / 5
Xsv = (Xv-dsmean) / dsstd
loader = DataLoader(TensorDataset(torchify(Xs), torchify(labels, torch.long)), batch_size=8, shuffle=True)
loader_v = DataLoader(TensorDataset(torchify(Xsv), torchify(labels_v, torch.long)), batch_size=256, shuffle=False)

from toolkit.networks import ConvolutionalAutoEncoder, count_parameters, AutoEncoder
import torch.nn as nn
import torch

class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, in_features, out_classes, complexity=2, latent_space=16, dropout=0.2):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.complexity = complexity
        self.in_features = in_features
        self.conv = nn.Sequential(
            nn.Conv1d(1, complexity*1, 7, 6),
            nn.BatchNorm1d(complexity*1),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(complexity*1, complexity*2, 5, 4),
            #nn.BatchNorm1d(complexity*2),
            nn.Tanh(),
            nn.Dropout1d(dropout),
        )
        
        self.deconv = nn.Sequential(
            #nn.ConvTranspose1d(complexity*4, complexity*2, 5, 2),
            #nn.ReLU(),
            nn.ConvTranspose1d(complexity*2, complexity*1, 5, 4),
            nn.BatchNorm1d(complexity*1),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.ConvTranspose1d(complexity*1, 1, 7, 6),
            #nn.BatchNorm1d(1),
            nn.Tanh(),
            nn.Dropout1d(dropout),
            nn.Upsample(in_features)
        )

        self.latent_size = self.conv(torch.randn(1, 1, in_features)).shape[-1]
        latent_size = self.latent_size
        self.l1 = nn.Linear(complexity*2*latent_size, latent_space)
        self.l2 = nn.Linear(latent_space, complexity*2*latent_size)
        self.lc = nn.Linear(complexity*2*latent_size, out_classes)

    def forward(self, X):
        code, classif = self.encode(X)
        return self.decode(code), torch.log_softmax(classif, dim=1)
    def encode(self, X):
        bs = X.shape[0]
        X1 = self.conv(X)
        X = torch.tanh(self.l1(X1.reshape(bs, -1)))
        return X, self.lc(X1.reshape(bs, -1))
    def decode(self, X):
        bs = X.shape[0]
        X = torch.relu(self.l2(X).reshape(bs, self.complexity*2, -1))
        X = self.deconv(X)
        return X
ae = ConvolutionalAutoEncoder(len(wavelength), out_classes=3, complexity=6, latent_space=2, dropout=0.5)
print(count_parameters(ae))
print(ae.latent_size)


from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
best_model = None
best_model_valid = 0.0
ae.train()
optim = Adam(ae.parameters(), lr=3e-5, weight_decay=1e-1)
#scheduler = StepLR(optim, step_size=10, gamma=0.95)
mse_loss = nn.MSELoss()
nll_loss = nn.NLLLoss()
losses = list()
losses_t = list()
accs = list()
for e in range(400):
    epoch_loss = 0.0
    ae.train()
    for x, y  in loader:
        bs = x.shape[0]
        ds = x.shape[1]
        x = ((((x * dsstd) + dsmean) * (0.95 + 0.1 * torch.rand(bs, 1))) - dsmean) / dsstd
        x = x.type(torch.float)
        #x = torch.from_numpy(scaler.inverse_transform(5*x) * (0.95 + 0.1 * np.random.rand(bs, 1)))
        #linear_deform = torch.randn(bs, 1) * (torch.from_numpy(wavelength)-(np.max(wavelength)-np.min(wavelength)/2)) * 1e-4
        
        #x += linear_deform
        #x = torch.from_numpy(1/5*scaler.transform(x)).type(torch.float)
        X_noisy = x + torch.randn(bs, ds) * 0.1
        X_noisy = X_noisy.reshape(bs, 1, -1)
        x = x.reshape(bs, 1, -1)
        optim.zero_grad()
        xr, yp = ae(X_noisy)
        recons_loss = mse_loss(xr, x)
        classif_loss = nll_loss(yp, y)
        loss = classif_loss + recons_loss
        
        loss.backward()
        optim.step()
        epoch_loss += loss.cpu().item()

    with torch.no_grad():
        ae.eval()
        epoch_loss_t = 0.0
        mean_acc = 0.0
        for x, y in loader_v:
            bs = x.shape[0]
            xp, yp = ae(x.reshape(bs, 1, -1))
            loss = nll_loss(yp, y)
            epoch_loss_t += loss.item()
            mean_acc += np.count_nonzero(np.argmax(yp.detach().numpy(), 1) == y.detach().numpy()) / len(y)
        epoch_loss_t /= len(loader_v)
        mean_acc /= len(loader_v)
        losses_t.append(epoch_loss_t)
        accs.append(mean_acc)

    if mean_acc >= best_model_valid:
        best_model = ae.state_dict().copy()
        best_model_valid = mean_acc


    epoch_loss /= len(loader)
    print(f"epoch {e} loss {round(epoch_loss, 4)} valid {round(epoch_loss_t, 5)}, acc {round(mean_acc, 3)} ")#lr {scheduler.get_last_lr()[0]:.1g}")
    losses.append(epoch_loss)

ae.load_state_dict(best_model)

torch.save({"model": ae.state_dict(), "optim": optim }, "model_004.pkl")

import shap
import torch
import torch.nn as nn
ae.eval()
class Classifier(nn.Module):
    def __init__(self, ae):
        super(Classifier, self).__init__()
        self.ae = ae
        
    def forward(self, X):
        _, classif = self.ae(X)
        return classif

classif_model = Classifier(ae)
batch = torch.from_numpy(Xs).type(torch.float).reshape(-1, 1, len(wavelength))

background = batch[:]
test_images = batch[:]
print(test_images.shape)

e = shap.DeepExplainer(classif_model, background)
shap_values = e.shap_values(test_images)
shap_values = np.asarray(shap_values).copy()
test_images=test_images[:, 0]

np.savez_compressed(f"shaps_{args.num:03}.npz", test_images=test_images, shap_values=shap_values)