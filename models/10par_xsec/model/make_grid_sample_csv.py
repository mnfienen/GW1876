import numpy as np
import pandas as pd
import pyemu

nsamples = 5

hk_values = np.linspace(0.1,5.0,nsamples)
fx_values = np.linspace(0.1,2.0,nsamples)

with open("grid.csv",'w') as f:
	f.write("hk1,cal_flux\n")
	for v1 in hk_values:
		for v2 in fx_values:
			f.write("{0:15.6E},{1:15.6E}\n".format(v1,v2))








