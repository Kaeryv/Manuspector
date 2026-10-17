from glob import glob
from dataset import read_asc
from os.path import basename
#references = ["DG03A", "DG05A", "DE55E", "OE17D" ]
references = ["9387", "9238", "9566", "9667" ]
colors     = ["#000000", "#888888", "#AAAAAA", "#DDDDDD" ]

'''
files = glob("./raw_data/DataJulie/Refs/*.Sample.asc")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for file in files:
    f = basename(file)[:5]
    index = references.index(f)
    print(f)
    d = read_asc(file)
    d = d[1]
    if "sans" in file:
        ax.plot(d, color=colors[index])
    else:
        ax.plot(d, color=colors[index], ls=":")

plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np
files = glob("./raw_data/DataJulie/Refs/001-*.Sample.asc")
print(len(files))
fig, ax = plt.subplots()
db = np.zeros((4, 206))
for file in files:
    f = basename(file)[19:19+4]
    index = references.index(f)
    d = read_asc(file)
    d = d[1]
    if "JB02" in file:
        ax.plot(d, color=colors[index], ls=":")
    else:
        db[index] = d
        ax.plot(d, color=colors[index])

d2 = db  
plt.show()
fig, ax = plt.subplots()
ax.plot(d2[0]**2.5/6/6/6)
ax.plot(d2[-1])
plt.show()

