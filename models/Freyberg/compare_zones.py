import os
import numpy as np
import matplotlib.pyplot as plt

z1 = np.loadtxt(os.path.join("Freyberg_zones","hk.zones"))
z2 = np.loadtxt(os.path.join("Freyberg_Truth","kzone.ref"))

fig = plt.figure(figsize=(10,10))

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1.imshow(z1,interpolation="nearest")
ax2.imshow(z2,interpolation="nearest")

plt.show()