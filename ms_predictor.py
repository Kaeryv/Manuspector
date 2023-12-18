from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-savgol", nargs=2, type=int, default=[11, 2]) # Savgol kernel size and order.
parser.add_argument("-max_samples", default=3, type=int)
parser.add_argument("-validation", default="historical", type=str)
parser.add_argument("-bs", type=int, default=8)
parser.add_argument("-model_file", type=str, default="default.model")
parser.add_argument("-chart_file", type=str, default="default.png")
parser.add_argument("-epochs", type=int, default=500)
parser.add_argument("-rho", type=float, default=1.0)
parser.add_argument("-complexity", type=int, default=6)
parser.add_argument("-latent", type=int, default=2)
parser.add_argument("-folds", type=int, default=[4, 0], nargs=2)
parser.add_argument("-dropout", type=float, default=0.5)
parser.add_argument("-cwt", type=int, nargs=4, default=None) #[35, 40, 1, 4]
parser.add_argument("-lr", type=float, default=3e-5)
parser.add_argument("-wd", type=float, default=1e-1)
parser.add_argument("-seed", type=int, default=42)
parser.add_argument("-noise", type=float, default=0.0)
parser.add_argument("-action", type=str, default="train")
args = parser.parse_args()

import torch
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import numpy as np
from misc import data_augment, count_parameters, torchify
from dataset import load_spectral_data, dataset_dict_to_dense

np.random.seed(args.seed*543246)


dev = torch.device("cpu")

import torch

    


# Training data
from scipy.signal import savgol_filter
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


if args.action =="savgol":
    Xr = X.copy()
if args.savgol[0] > 0:
    X = savgol_filter(X, args.savgol[0], args.savgol[1], axis=-1)

if args.action =="savgol":
    fig, ax = plt.subplots()
    ax.plot(Xr[0], label="RAW")
    ax.plot(X[0], label=f"Savgol w={args.savgol[0]} k={args.savgol[1]}")
    plt.legend()
    plt.show()
    del Xr


if args.cwt is not None:
    from scipy.signal import cwt, ricker
    def do_cwt(data, octaves=128, offset=1, stridesf=1, stridesx=1):
        cwts = list()
        for i in range(len(data)):
            cwts.append(cwt(data[i], wavelet=ricker, widths=np.arange(offset+1, offset+octaves+1)))
        cwts = np.asarray(cwts) / 5
        return np.abs(cwts[:, ::stridesf, ::stridesx]).real #[:, offset:offset+octaves]

    X = do_cwt(X, octaves=args.cwt[0], offset=args.cwt[1], stridesf=args.cwt[2], stridesx=args.cwt[3])

# StandardScaling
dsmean = X.mean(axis=0)
dsstd = X.std(axis=0) * 2
X = (X - dsmean) / dsstd

if args.validation == "historical":
    # Separate every single historic parchment and put them in validation (sets 12 and 13).
    valid_mask = np.logical_or(Y[:, 0] == 13, Y[:, 0] == 12)
    # Train/Test split
    Xv = X[valid_mask]
    Yv = Y[valid_mask]
    Xt = X[~valid_mask]
    Yt = Y[~valid_mask]

elif args.validation == "historical+":
    # Separate every single historic parchment and put them in validation (sets 12 and 13).
    valid_mask = np.logical_or(Y[:, 0] == 13, Y[:, 0] == 12)
    # Train/Test split
    Xv = X[valid_mask]
    Yv = Y[valid_mask]
    Xt = X.copy()
    Yt = Y.copy()
elif args.validation == "segment":
    # will be used for k-fold
    validation_ids = np.arange(len(Y))
    np.random.shuffle(validation_ids)
    validation_ids = np.array_split(validation_ids, args.folds[0])
    valid_mask = np.zeros(len(Y)).astype(bool)
    valid_mask[validation_ids[args.folds[1]]] = True
    Xv = X[valid_mask]
    Yv = Y[valid_mask]
    Xt = X[~valid_mask]
    Yt = Y[~valid_mask]

if args.cwt:
    img_shape = (X.shape[1],X.shape[2])
del X, Y

# Limit samples per parchment
Xt = Xt[Yt[:,5]< args.max_samples,:]
Yt = Yt[Yt[:,5]< args.max_samples,:]

Xv = Xv[Yv[:,5]< 4,:]
Yv = Yv[Yv[:,5]< 4,:]

labels_t = Yt[:, 1]
labels_v = Yv[:, 1]

print("Samples per class: ", [np.count_nonzero(labels_t == i) for i in range(3)])


from torch.utils.data import TensorDataset, DataLoader
import torch

loader_t = DataLoader(TensorDataset(torchify(Xt, cc=1), torchify(labels_t, torch.long), torchify(Yt[:, 0], torch.long)), batch_size=args.bs, shuffle=True)
loader_v = DataLoader(TensorDataset(torchify(Xv, cc=1), torchify(labels_v, torch.long)), batch_size=256,     shuffle=False)


import torch.nn as nn
import torch

import torch.nn as nn
class WaveletAutoEncoder(nn.Module):
    def __init__(self, complexity, dropout) -> None:
        super(WaveletAutoEncoder, self).__init__()
        c = complexity
        self.conv = nn.Sequential(
            nn.Conv2d(1    , 1 * c, 5, 2),
            nn.LeakyReLU(0.0, True),
            nn.Dropout2d(dropout),
            nn.Conv2d(1 * c, 2 * c, 5, (1,2)),
            nn.LeakyReLU(0.0, True),
            nn.Conv2d(2 * c, 4 * c, 5, (1,2)),
            nn.Tanh(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, 5, (1,2)),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(c * 2, c * 1, 5, (1,2)),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(c * 1,  1, 5, 2),
            nn.Upsample(img_shape)
        )
        size = self.conv(torch.randn(1,1, img_shape[0], img_shape[1])).shape
        self.dense = nn.Sequential(
            nn.Linear(complexity*4*size[2]*size[3], 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 3),
            nn.LogSoftmax(1)
        )
    
    def forward(self, X):
        bs = X.shape[0]
        X = self.conv(X)
        C = self.dense(X.reshape(bs, -1))
        Y = self.deconv(X)
        return Y, C



class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, in_features, out_classes, complexity=2, latent_space=16, dropout=0.2, kernel_base=5):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.complexity = complexity
        self.in_features = in_features
        self.conv = nn.Sequential(
            nn.Conv1d(1, complexity*1, kernel_base+2, kernel_base+1),
            nn.ReLU(),
            #nn.Dropout1d(dropout),
            nn.Conv1d(complexity*1, complexity*2, kernel_base, kernel_base-1),
            nn.Tanh(),
            nn.Dropout1d(dropout),
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(complexity*2, complexity*1, kernel_base, kernel_base-1),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.ConvTranspose1d(complexity*1, 1, kernel_base+2, kernel_base+1),
            nn.Tanh(),
            #nn.Dropout1d(dropout),
            nn.Upsample(in_features)
        )

        latent_size = self.conv(torch.randn(1, 1, in_features)).shape[-1]
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

if args.cwt is None:
    ae = ConvolutionalAutoEncoder(len(wavelength), out_classes=3, complexity=args.complexity, latent_space=args.latent, dropout=args.dropout)
else:
    ae = WaveletAutoEncoder(complexity=args.complexity, dropout=args.dropout)
ae.to(dev)
print("PARAMETERS", count_parameters(ae))

from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import trange

optim = Adam(ae.parameters(), lr=args.lr, weight_decay=args.wd)

def accuracy(prediction, ground_truth):
    return np.count_nonzero(np.argmax(prediction, 1) == ground_truth) / len(ground_truth)

def train(model, optim, num_epochs, loader_t, loader_v, noise=None, dont_train_classif_on_historic=True):
    best_model = None
    best_model_valid = 0.0
    mse_loss = nn.MSELoss()
    nll_loss = nn.NLLLoss()
    progress = trange(num_epochs)
    # Monitoring
    losses = list()
    losses_t = list()
    accs = list()
    for e in progress:
        epoch_loss = 0.0
        model.train()
        for x, y, z  in loader_t:
            bs = x.shape[0]
            ds = x.shape[1]
            if noise is not None:
                X_noisy = torchify(data_augment(x.numpy()), cc=1)
                X_noisy = x + torch.randn(bs, 1, ds) * noise
                X_noisy = X_noisy.reshape(bs, 1, -1)
                x = x.reshape(bs, 1, -1)
            else:
                X_noisy = x
                
            optim.zero_grad()
            xr, yp = model(X_noisy.to(dev))
            recons_loss = mse_loss(xr.cpu(), x)
            if dont_train_classif_on_historic:
                is_historical = torch.logical_or(z==12, z==13)
                if torch.all(is_historical):
                    classif_loss = 0.0
                else:
                    classif_loss = nll_loss(yp.cpu()[~is_historical], y[~is_historical])
            else:
                classif_loss = nll_loss(yp.cpu(), y)
            loss = (1-args.rho) * classif_loss + args.rho * recons_loss
            
            loss.backward()
            optim.step()
            epoch_loss += loss.cpu().item()
        with torch.no_grad():
            model.eval()
            epoch_loss_t = 0.0
            mean_acc = 0.0
            for x, y in loader_v:
                bs = x.shape[0]
                xp, yp = ae(x.to(dev))
                loss = nll_loss(yp.cpu(), y)
                epoch_loss_t += loss.item()
                mean_acc += accuracy(yp.cpu().detach().numpy(), y.detach().numpy())
            epoch_loss_t /= len(loader_v)
            mean_acc /= len(loader_v)
            losses_t.append(epoch_loss_t)
            accs.append(mean_acc)

        if mean_acc >= best_model_valid:
            best_model = ae.state_dict().copy()
            best_model_valid = mean_acc


        epoch_loss /= len(loader_t)
        progress.set_description(f"epoch {e} loss {round(epoch_loss, 4)} valid {round(epoch_loss_t, 5)}, acc {round(mean_acc, 3)} ")
        losses.append(epoch_loss)

    ae.load_state_dict(best_model)

    return losses, losses_t, accs, best_model_valid


if args.action == "train":
    print("Training model")
    losses, losses_t, accs, best_model_valid = train(ae, optim, args.epochs, loader_t, loader_v, noise=args.noise if args.cwt is None else None)
    print(f"MEDACC: {np.median(accs)}")


    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(losses, label="Training loss")
    ax1.set_ylabel("T. Classif+Recons loss")
    ax1b = ax1.twinx()
    ax1b.set_ylabel("V. Classif loss")
    ax1b.plot(losses_t, label="Validation loss")
    ax2.plot(accs, color="g", label="Accuracy")
    ax2.axhline(best_model_valid, color="green")
    print(np.min(losses_t))
    ax2.set_ylim(0, 1)
    plt.savefig(args.chart_file)
    torch.save({
        "model": ae.state_dict(), 
        "optim": optim, 
        "losses": losses, 
        "accs": accs, 
        "losses_t": losses_t,
        "num_parameters": count_parameters(ae),
        "seed": torch.seed()
    }, args.model_file)
    print(torch.seed())

elif args.action == "augment":
    fig, axs = plt.subplots()
    #axs = axs.flatten()
    axs.plot(Xt[0])
    for i in range(5):
        xaug = data_augment(Xt[:2])
        xaug = xaug.reshape(2, 1, xaug.shape[-1])
        xaug += np.random.randn(2, 1, xaug.shape[-1]) * args.noise
        axs.plot(xaug[0,0])
    plt.show()

