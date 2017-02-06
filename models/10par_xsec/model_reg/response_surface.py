import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyemu

nsamples = 100
hk_values = np.linspace(0.1,4.0,nsamples)
fx_values = np.linspace(.1,2.5,nsamples)

def write_input_csv():
	c = 0
	with open("grid.csv",'w') as f:
		f.write("run_id,hk1,cal_flux\n")
		for v1 in hk_values:
			for v2 in fx_values:
				f.write("{0:d},{1:15.6E},{2:15.6E}\n".format(c,v1,v2))
				c += 1

def run_sweep():
	os.system("sweep k_wel_reg .pst")
	#os.chdir("..")
	#pyemu.pst_utils.start_slaves(slave_dir="model",
	#	exe_rel_path="sweep",num_slaves=5,pst_rel_path="k_wel.pst",port=4005)
	#os.chdir("model")

def make_surface_plot():
	df_in = pd.read_csv("grid.csv")
	df_out = pd.read_csv("sweep_out.csv")
	resp_surf = np.zeros((nsamples,nsamples))
	c = 0
	for i,v1 in enumerate(hk_values):
		for j,v2 in enumerate(fx_values):
			resp_surf[j,i] = df_out.loc[c,"phi"]
			c += 1
	fig = plt.figure(figsize=(10,10))
	ax = plt.subplot(111)
	X,Y = np.meshgrid(hk_values,fx_values)
	resp_surf = np.ma.masked_where(resp_surf>5,resp_surf)
	p = ax.pcolor(X,Y,resp_surf,alpha=0.5,cmap="spectral")
	plt.colorbar(p)
	c = ax.contour(X,Y,resp_surf,levels=[0.1,0.2,0.5,1,2,5],colors='k')
	plt.clabel(c)
	ax.set_xlim(hk_values.min(),hk_values.max())
	ax.set_ylim(fx_values.min(),fx_values.max())
	ax.set_xlabel("hk1 ($\\frac{L}{T}$)")
	ax.set_ylabel("cal_flux ($\\frac{L^3}{T}$)")
	plt.savefig("resp_surf.png")

if __name__ == "__main__":
	#write_input_csv()
	#run_sweep()
	make_surface_plot()






