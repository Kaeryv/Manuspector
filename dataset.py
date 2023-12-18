import os
import numpy as np
from glob import glob
from os.path import basename


species = { "C": 0, "S": 1, "G": 2}
type = { "C": 0, "F": 1, "X": 2, "Y": 3}
location = { "H10": 0, "N10": 1, "XXX": 2, "XXY": 2, "YYX": 2}

def read_asc(filename, no_header=False):
    ''' Read an asc file to numpy spectrum 
    :parameter filename: The path of the file
    :parameter no_header: Boolean flag telling if the asc header is present (#DATA)
    :return: A numpy array of shape (2, N) where N is the number of points.
    '''

    skip = True
    if no_header:
        skip = False
        
    text_data = list()
    if not os.path.isfile(filename):
        print(f"File {filename} does not exist!")
        exit()
    with open(filename, "r") as f:
        for line in f.readlines():
            if skip:
                if line == "#DATA\n":
                    skip = False
                continue
            
            text_data.append(line)
    if skip:
        print(f"No data while reading {filename}")
        exit()

    res = np.fromstring("".join(text_data), dtype=float, sep='\t')
    return res.reshape((-1, 2)).T

def load_spectral_data(paths, verbose=False):
    '''
        Loads spectra dataset from list of paths.
    '''
    loaded_data = dict()
    for path in paths:
        files_list = glob(path, recursive=True)
        if verbose:
            print(f"Loading {len(files_list)} files from {path}")
        for filepath in files_list:
            sample_name = basename(filepath)
            sample_name = sample_name.replace(".Sample.Raw", "").replace(".ASC", ".asc")

            data = read_asc(filepath)
            loaded_data[sample_name] = data[1]

    return loaded_data

def dataset_dict_to_dense(dataset: dict):
    '''
    Creates the six labels for each spectrum.
    '''
    len_data = len(dataset)
    len_wavelength = len(list(dataset.values())[0])
    X = np.zeros((len_data, len_wavelength))
    Y = np.zeros((len_data, 6), dtype=np.uint32)
    for i, (fname, spectrum) in enumerate(dataset.items()):
        X[i, :] = spectrum
        Y[i, 0] = int(fname[0:3])        # Skin
        Y[i, 1] = species[fname[5]]      # Espèce
        Y[i, 2] = int(fname[4])          # Process step
        Y[i, 3] = location[fname[7:10]]  # Body Location
        Y[i, 4] = type[fname[11:12]]     # Grain / Flesh
        Y[i, 5] = int(fname[12:15])      # ID
        if fname[7:10] not in location.keys():
            print("ERROR")
        if fname[11:12] not in type.keys():
            print("ERROR")
        if fname[5] not in species.keys():
            print("ERROR")
    return X, Y.astype(np.int32)
