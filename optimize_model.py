from hybris.optim import ParticleSwarm
from keever.runners import generate_job
import numpy as np
from copy import copy

opt_seed = 42
seeds = 3

def read_output():
    file = "out.txt"
    with open("out.txt") as f:
        for line in f.readlines():
            if "MEDACC" in line:
                return float(line.split(" ")[1])



with open("launch.proto.sh", "r") as f:
    template = f.read()

hyperparameters = {
        "lr": [ 1e-5, 1e-3, float ],
        "rho": [0.0, 1.0, float ],
        "noise": [1e-4, 1e-3, float],
        "complexity": [ 2.0, 8.0, int],
        "latent": [2.0, 10.0, int],
        "offset": [ 16, 32, int],
        "dropout": [0.0, 0.5, float]
    }

boundaries = np.asarray([ tuple(map(float, (value[0], value[1]))) for key, value in hyperparameters.items() ])
ND = len(hyperparameters)
opt = ParticleSwarm(20, [ND, 0], max_fevals=4000)
opt.weights[0] = 0.5
opt.weights[6] = 0.5
opt.vmin = boundaries[:, 0]
opt.vmax = boundaries[:, 1]
opt.reset(opt_seed)


#sample = opt.vmin + np.random.rand((ND)) * (opt.vmax-opt.vmin)

def evaluate_config(sample):
    config_accs = list()
    config = { 
        key: typefun[2](value) 
        for key, value, typefun in zip(hyperparameters.keys(), sample, hyperparameters.values()) 
    }
    for s in range(seeds):
        config.update({"seed": s})
        generate_job(copy(template), config, launch=True)
        config_accs.append(read_output())
    
    fitness = np.mean(config_accs)
    return fitness

while not opt.stop():
    x = opt.ask()
    y = np.asarray([ evaluate_config(xx) for xx in x ])
    opt.tell(y)
