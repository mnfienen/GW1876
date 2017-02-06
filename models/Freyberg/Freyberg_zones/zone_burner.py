import numpy as np

indat = np.genfromtxt('hk_vals.dat', dtype=None, names=True)

indat = dict(zip([i.decode() for i in indat['param']],indat['val']))

inzones = np.loadtxt('hk.zones', dtype=int)
outzones = inzones.copy().astype(np.float32)

allzones = np.unique(inzones)

for cz in allzones:
    outzones[outzones==cz] = indat['hk{0}'.format(cz)]

np.savetxt('hk.ref', outzones, fmt = '%10.6f')
