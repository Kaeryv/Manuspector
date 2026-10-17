from glob import glob
from os.path import join, basename, dirname
from shutil import move
files = glob("raw_data/DataDavid/UVVIS/*.asc", recursive=True)
print("Detected", len(files), "files.")

for filepath in files:
    filename = basename(filepath)
    tokens = filename.split("-")
    tokens.insert(3, "DG001")
    new_filename = "-".join(tokens)
    new_filepath = filepath.replace(filename, new_filename)
    move(filepath, new_filepath)
    print(new_filepath)
