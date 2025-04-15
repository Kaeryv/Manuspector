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
parser.add_argument("-wlmin", type=float, default=200)
parser.add_argument("-wlmax", type=float, default=2300)
parser.add_argument("-adda", action="store_true")
parser.add_argument("-adda_complexity", type=int, default=128)
parser.add_argument("-adda_wd", type=float, default=1e-2)
args = parser.parse_args()


from copy import deepcopy
import torch
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import numpy as np
from scipy.signal import savgol_filter
from misc import data_augment, count_parameters, torchify
from dataset import load_spectral_data, dataset_dict_to_dense, augment_mix
from pywt import cwt
np.random.seed(args.seed*543246)


dev = torch.device("cpu")


validation_indices = list()
# Training data
with open("data.txt") as f:
    sources = f.read().splitlines()
    # Find historical samples
    for s in sources:
        if "VALIDATION" in s:
            validation_indices = list(map(int, s.split()[1:]))
    sources = [ s for s in sources if  not s.startswith("#")]
    dataset_dict = load_spectral_data(sources, verbose=True)

names = list(dataset_dict.keys())
# We label the dataset and convert it to numpy arrays
X, Y = dataset_dict_to_dense(dataset_dict)

# Build the wavelength vector
wavelength = np.flipud(np.linspace(250, 2500, 226))
# Wavelengths above 2300nm are removed because of low SNR
wl_mask = (wavelength <= args.wlmax) & (wavelength >= args.wlmin)
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


#X, Y =  augment_mix(X, Y, added=100)

if args.cwt is not None:
    def do_cwt(data, octaves=128, offset=1, stridesf=1, stridesx=1):
        import pywt
        cwts = list()
        for i in range(len(data)):
            cwts.append(cwt(data[i], np.arange(offset+1, offset+octaves+1), "gaus3")[0])
        cwts = np.asarray(cwts) / 5
        return np.abs(cwts[:, ::stridesf, ::stridesx]).real
    X1D = X.copy()
    X = do_cwt(X, octaves=args.cwt[0], offset=args.cwt[1], stridesf=args.cwt[2], stridesx=args.cwt[3])


# StandardScaling
dsmean = X.mean(axis=(0))
dsstd = X.std(axis=(0)) * 2
X = (X - dsmean) / dsstd

if args.action =="show_cwt":
    from scipy.ndimage import gaussian_filter
    fig, axs = plt.subplots(3, 3, figsize=(6.2, 4/2*3), dpi=150, sharex=True)
    for i in range(3):
        mask = Y[:, 1] == i
        axs[0,i].matshow(np.flip(X[mask][0], axis=-1), cmap="RdBu", vmin=-1, vmax=1, extent=[wavelength[-1], wavelength[0], args.cwt[1],args.cwt[1]+args.cwt[0]], origin="lower")
        axs[1,i].matshow(gaussian_filter(np.flip(X[mask][0], axis=-1), 2), cmap="RdBu", vmin=-1, vmax=1, extent=[wavelength[-1], wavelength[0], args.cwt[1],args.cwt[1]+args.cwt[0]], origin="lower")
        axs[0,i].set_aspect(20)
        axs[1,i].set_aspect(20)
        axs[2,i].plot(np.flipud(wavelength), np.flipud(X1D[mask][0]))
    plt.tight_layout()
    plt.savefig("cwt_illus.png")
    plt.savefig("cwt_illus.pdf")

histo_mask = np.zeros_like(Y[:, 0], dtype=bool)
for v in validation_indices:
    histo_mask |= (Y[:, 0] == v)

use_histo_in_classif = False

if args.validation == "historical":
    # Separate every single historic parchment and put them in validation (sets 12 and 13).
    valid_mask = histo_mask
    # Train/Test split
    Xv = X[valid_mask]
    Yv = Y[valid_mask]
    Xt = X[~valid_mask]
    Yt = Y[~valid_mask]
elif args.validation == "all":
    use_histo_in_classif=True
    Xv = X.copy()
    Yv = Y.copy()
    Xt = X.copy()
    Yt = Y.copy()
    valid_mask = np.ones_like(histo_mask)

elif args.validation == "historical+":
    # Separate every single historic parchment and put them in validation (sets 12 and 13).
    valid_mask = histo_mask
    # Train/Test split
    Xv = X[valid_mask]
    Yv = Y[valid_mask]
    Xt = X.copy()
    Yt = Y.copy()
elif args.validation == "segment":
    use_histo_in_classif=True
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

names_validation = list()
names_training = list()
for i, is_validaton in enumerate(valid_mask):
    if is_validaton:
        names_validation.append(names[i])
        if args.validation == "historical+":
            names_training.append(names[i])
    else:
        names_training.append(names[i])

if args.cwt:
    img_shape = (X.shape[1],X.shape[2])
#del X, Y

# Limit samples per parchment
sample_limiter = Yt[:,5]< args.max_samples
Xt = Xt[sample_limiter,:]
Yt = Yt[sample_limiter,:]

if len(names_training) > 0:
    names_training = [ names_training[i] for i, is_included in enumerate(sample_limiter) if is_included ]

sample_limiter = Yv[:,5]< 4
Xv = Xv[sample_limiter,:]
Yv = Yv[sample_limiter,:]

names_validation = [ names_validation[i] for i, is_included in enumerate(sample_limiter) if is_included ]


labels_t = Yt[:, 1]
labels_v = Yv[:, 1]


print("Samples per class [t]: ", [np.count_nonzero(labels_t == i) for i in range(3)])
print("Samples per class [v]: ", [np.count_nonzero(labels_v == i) for i in range(3)])


from torch.utils.data import TensorDataset, DataLoader
import torch

loader_t = DataLoader(TensorDataset(torchify(Xt, cc=1), torchify(labels_t, torch.long), torchify(Yt[:, 0], torch.long)), batch_size=args.bs, shuffle=True)
loader_v = DataLoader(TensorDataset(torchify(Xv, cc=1), torchify(labels_v, torch.long)), batch_size=256,     shuffle=False)


import torch.nn as nn
import torch
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, ae) -> None:
        super(Discriminator, self).__init__()
        complexity = ae.complexity
        size = ae.size
        self.seq = nn.Sequential(
            nn.Linear(complexity*4*size[2]*size[3], args.adda_complexity),
            nn.ReLU(),
            nn.Linear(args.adda_complexity, 1),
            nn.Sigmoid()
        )
    def forward(self, X):
        bs = X.shape[0]
        X = X.reshape(bs, -1)
        return self.seq(X)
import torch.nn as nn
class WaveletAutoEncoder(nn.Module):
    def __init__(self, complexity, dropout) -> None:
        super(WaveletAutoEncoder, self).__init__()
        c = complexity
        self.complexity = c
        self.conv = nn.Sequential(
            # Layer 1
            # Maxpool smaller : SMP old = 5 3 3
            nn.Conv2d(1, 1 * c, 5, padding="valid"),
            nn.MaxPool2d(2, (1, 2)),
            nn.Tanh(),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(1*c),
            # Layer 2
            nn.Conv2d(1 * c, 2 * c, 3, padding="valid"),
            nn.MaxPool2d(2, (1,2)),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(2*c),
            # Layer 3
            nn.Conv2d(2 * c, 4 * c, 3, padding="valid"),
            nn.MaxPool2d(2, (1,2)),
            nn.Tanh(),
        )
        self.deconv = nn.Sequential(
            # Layer 1
            nn.Upsample(scale_factor=(1,2)),
            nn.Conv2d(c * 4, c * 2, 3, 1),
            nn.Tanh(),
            # Layer 2
            nn.Upsample(scale_factor=(1,2)),
            nn.Conv2d(c * 2, c * 1, 3, 1),
            nn.Tanh(),
            # Layer 3
            nn.Upsample(scale_factor=(1,2)),
            nn.Conv2d(c * 1,  1, 5, 1),
            nn.Upsample(img_shape)
        )
        size = self.conv(torch.zeros(1,1, img_shape[0], img_shape[1])).shape
        self.size = size
        self.dense = nn.Sequential(
            nn.Linear(complexity*4*size[2]*size[3], 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            #nn.Softmax(1)
        )

        
    
    def forward(self, X):
        bs = X.shape[0]
        X = self.conv(X)
        C = self.dense(X.reshape(bs, -1))
        Y = self.deconv(X)
        return Y, F.log_softmax(C, 1), X



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
        is_histo = self.discriminator(X)
        print(is_histo)
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
dc = Discriminator(ae).to(dev)

from torch.optim import Adam, SGD
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import trange
from copy import copy


def count_correct(prediction, ground_truth):
    return np.count_nonzero(np.argmax(prediction, 1) == ground_truth)

def train(model, num_epochs, loader_t, loader_v, noise=None, dont_train_classif_on_historic=True):
    optim = SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optim_dc = Adam(dc.parameters(), lr=5e-4, weight_decay=args.adda_wd)
    best_model = None
    best_model_valid = 0.0
    mse_loss = nn.MSELoss()
    nll_loss = nn.NLLLoss()
    bce_loss = nn.BCELoss()
    progress = trange(num_epochs)
    # Monitoring
    losses = list()
    losses_t = list()
    accs = list()
    losses_dc = list()
    for e in progress:
        epoch_loss = 0.0
        epoch_recons_loss = 0.0
        epoch_dc_loss = 0.0
        model.train()
        for x, y, z  in loader_t:
            bs = x.shape[0]
            ds = x.shape[1]
            if args.noise > 0.0:
                X_noisy = x * (1.0 + torch.randn(1)*0.5)
                X_noisy = X_noisy + torch.randn(x.shape) * 0.05
            else:
                X_noisy = x
            optim.zero_grad()
            xr, yp, lat = model(X_noisy.to(dev))
            recons_loss = mse_loss(xr.cpu(), x)
            if dont_train_classif_on_historic:
                is_historical = torch.zeros_like(z).type(torch.bool)
                for v in validation_indices:
                    is_historical |= (z==v)
                if torch.all(is_historical):
                    classif_loss = 0.0
                else:
                    classif_loss = nll_loss(yp.cpu()[~is_historical], y[~is_historical])
                optim_dc.zero_grad()
                dc_loss = bce_loss(dc(lat.detach()), is_historical.float().reshape(bs, 1))
                dc_loss.backward()
                epoch_dc_loss += dc_loss.cpu().item()
                optim_dc.step()
            else:
                classif_loss = nll_loss(yp.cpu(), y)
            histo_loss = bce_loss(dc(lat), torch.zeros(bs, 1).to(dev))
            #print(histo_loss.cpu().item())
            loss = (1-args.rho) * classif_loss + args.rho * recons_loss + 2*histo_loss
            loss.backward()
            #print(loss.item())
            #exit()
            optim.step()
            epoch_loss += loss.cpu().item()
            epoch_recons_loss += recons_loss.cpu().item()
        with torch.no_grad():
            model.eval()
            epoch_loss_t = 0.0
            correct_samples = 0
            for x, y in loader_v:
                bs = x.shape[0]
                _, yp, _ = model(x.to(dev))
                loss = nll_loss(yp.cpu(), y)
                epoch_loss_t += loss.item()
                correct_samples += count_correct(yp.cpu().detach().numpy(), y.detach().numpy())
            epoch_loss_t /= len(loader_v)
            mean_acc = correct_samples / len(Yv)
            losses_t.append(epoch_loss_t)
            accs.append(mean_acc)

        if mean_acc >= best_model_valid:
            best_model = deepcopy(ae.state_dict())
            best_model_valid = copy(mean_acc)

        if e % 5 == 0:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(losses, label="Training loss", color="r")
            ax1.plot(losses_dc, label="Discr loss", color="g")
            ax1.set_ylabel("T. Classif+Recons loss")
            ax1b = ax1.twinx()
            ax1b.set_ylabel("V. Classif loss")
            ax1b.plot(losses_t, label="Validation loss", color="b")
            ax2.plot(accs, color="g", label="Accuracy" )
            ax2.axhline(np.max(accs), color="green")
            ax2.set_ylim(0, 1)
            plt.legend()
            plt.savefig(args.chart_file)


        epoch_dc_loss /= len(loader_t)
        losses_dc.append(epoch_dc_loss)
        epoch_loss /= len(loader_t)
        epoch_recons_loss /= len(loader_t)
        progress.set_description(f"ep. {e} loss {round(epoch_loss, 4)} valid {round(epoch_loss_t, 5)}, acc {round(mean_acc, 3)} dc_loss {round(epoch_dc_loss, 4)}")
        losses.append((epoch_loss, epoch_recons_loss))

    model.load_state_dict(best_model)

    return losses, losses_t, accs, best_model_valid, optim


if args.action == "train":
    print("Training model")
    losses, losses_t, accs, best_model_valid, optim = train(ae, args.epochs, loader_t, loader_v, noise=args.noise if args.cwt is None else None, dont_train_classif_on_historic=not use_histo_in_classif)
    print(f"MEDACC: {np.median(accs)}")
    print(f"MAXACC: {np.max(accs)}")
    print(f"ENDACC: {accs[-1]}")
    losses = np.asarray(losses)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(losses[:,0], label="Training loss", color="r")
    ax1.plot(losses[:,1], label="Recons loss", color="k")
    ax1.set_ylabel("T. Classif+Recons loss")
    ax1b = ax1.twinx()
    ax1b.set_ylabel("V. Classif loss")
    ax1b.plot(losses_t, label="Validation loss", color="b")
    ax2.plot(accs, color="g", label="Accuracy" )
    ax2.axhline(best_model_valid, color="green")
    print(np.min(losses_t))
    ax2.set_ylim(0, 1)
    plt.legend()
    plt.savefig(args.chart_file)
    torch.save({
        #"model": ae.state_dict(), 
        #"optim": optim, 
        "losses": losses[:,0], 
        "lrecons": losses[:,1], 
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

elif args.action == "shapley":
    print("Computing shapley values")
    import shap
    weights = torch.load(args.model_file)["model"]
    ae.load_state_dict(weights)
    print("Loaded model")
    def forward(X):
        X = do_cwt(X, octaves=args.cwt[0], offset=args.cwt[1], stridesf=args.cwt[2], stridesx=args.cwt[3])
        X = (X - dsmean) / dsstd
        _, yp = ae(torchify(X, cc=1).to(dev))
        return yp

    background = X1D[np.random.choice(len(X1D), size=min(150, len(X1D)))]
    print(background.shape)
    explainer = shap.explainers.Permutation(forward, background)
    print("SHAP Loaded")
    sample = np.random.choice(len(X1D), size=4)
    print(X1D[sample].shape)
    shap_values = explainer(X1D[sample])
    labels = Y[sample,1]
    print(shap_values.shape)
    np.savez("shap_values.npz", values=shap_values.values, base=shap_values.base_values, data=shap_values.data, labels=labels)


elif args.action == "shapley_cwt":
    print("Computing shapley values of CWT")
    import shap
    weights = torch.load(args.model_file)["model"]
    ae.load_state_dict(weights)
    print("Loaded model")
    downsampling = 1
    # Create a dummy model that only outputs classification
    class ClassifModel(nn.Module):
        def __init__(self, base) -> None:
            super(ClassifModel, self).__init__()
            self.base = base
            self.base.eval()
        def forward(self, X):
            print(X.shape)
            X = nn.Upsample(scale_factor=downsampling)(X)
            _, yp = self.base(X.to(dev))
            return yp #torch.exp(yp).clone()

    model = ClassifModel(ae)
    indices_background = np.random.choice(len(Xt), size=min(200, len(X1D)))
    background = torchify(Xt[indices_background],cc=1)
    background = nn.Upsample(scale_factor=1/downsampling)(background)
    
    # Using the SHAP DeepExplainer
    explainer = shap.explainers.DeepExplainer(model, background)
    sample = np.random.choice(len(background), size=100)
    Xshap = background[sample]
    shap_values = explainer.shap_values(Xshap,check_additivity=False)

    labels = Yt[sample,1]
    shap_values = np.asarray(shap_values)
    np.savez_compressed(f"shapley_data/shap_values_{args.seed}.npz", values=shap_values, data=Xshap.numpy(), labels=Yt[sample], spectra=X1D[sample], background=background[sample].numpy())



elif args.action == "predict":
    weights = torch.load(args.model_file)["model"]
    ae.load_state_dict(weights)
    ae.eval()
    #ae.conv[4].momentum = 0.0
    #ae.conv[4].track_running_stats=True
    #ae.conv[9].momentum = 0.0
    #ae.conv[9].track_running_stats=True
    #print(ae.conv[4])
    #print(ae.conv[9])
    #exit()
    val_spectra = torchify(Xv, cc=1)
    with torch.no_grad():
        mean_acc = 0.0
        xp, yp = ae(val_spectra.to(dev))
        preds = yp.detach().numpy()
        preds = np.argmax(preds, axis=1)
        print("ACC:", np.count_nonzero(preds==Yv[:,1])/len(Yv)*100, "%")
    
    with torch.no_grad():
        epoch_loss_t = 0.0
        correct_samples = 0
        for x, y in loader_v:
            bs = x.shape[0]
            xp, yp = ae(x.to(dev))
            correct_samples += count_correct(yp.cpu().detach().numpy(), y.detach().numpy())
        epoch_loss_t /= len(loader_v)
        mean_acc = correct_samples / len(Yv)
    print(mean_acc)
    exit()
    confusion_matrix = np.zeros((3,3))
    for pred, real in zip(preds, Yv[:,1]):
        confusion_matrix[pred, real] += 1
    import matplotlib.pyplot as plt
    from dataset import species
    fig, ax = plt.subplots()
    ax.matshow(confusion_matrix)
    for (i, j), z in np.ndenumerate(confusion_matrix):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    ax.set_xticks(np.arange(3), species)
    ax.set_yticks(np.arange(3), species)
    fig.savefig("Figures/confusion.png")
    histo_mask = Yv[:,0] == validation_indices[0]
    for v in validation_indices:
        histo_mask |= (Yv[:,0]==v)
    print("Historical samples count: ", np.count_nonzero(histo_mask))
    print("ACC:", np.count_nonzero(preds[histo_mask]==Yv[histo_mask,1])/np.count_nonzero(histo_mask)*100, "%")

    fig, axs = plt.subplots(10,5, figsize=(5,2))
    axs1 = axs.flatten()[:25]
    axs2 = axs.flatten()[25:]
    xp = xp.detach()[:len(axs1),0].numpy()
    vs = val_spectra.detach()[:len(axs),0].numpy()

    print("PREDVMIN", np.min(xp))
    print("PREDVMAX", np.max(xp))
    print("TRUEVMIN", np.min(vs))
    print("TRUEVMAX", np.max(vs))
    
    vmax = np.max(np.abs(vs))
    for i, ax in enumerate(axs1):
        ax.contourf(xp[i], vmin=-vmax, vmax=vmax)
        ax.axis("equal")
        ax.axis("off")
    plt.subplots_adjust(hspace=0.1)
    for i, ax in enumerate(axs2):
        ax.contourf(val_spectra.detach()[i,0].numpy(), vmin=-vmax, vmax=vmax)
        ax.axis("off")
    fig.savefig("Figures/reconstruction.png")
    fig.savefig("Figures/reconstruction.pdf")
elif args.action =="tsne":
    print("TSNEing")
    from sklearn.manifold import TSNE
    model = TSNE(2, perplexity=50)
    print(Xt.shape)
    Xtsne = model.fit_transform(Xt)
    import matplotlib.pyplot as plt
    histo_mask = Yt[:,0] == validation_indices[0]
    for v in validation_indices:
        histo_mask |= (Yt[:,0]==v)
    plt.scatter(*Xtsne[histo_mask].T, c=Yt[histo_mask,1])
    plt.scatter(*Xtsne[~histo_mask].T, c=Yt[~histo_mask,1], marker="+")
    plt.savefig("test2.png")
    print(Xtsne.shape)
