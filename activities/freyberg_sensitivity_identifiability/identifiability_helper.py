import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
import numpy as np

def plot_id_bars(ev, numsing):



    indf = ev.get_identifiability_dataframe(singular_value=numsing)

    [indf.drop(i, inplace=True) for i in indf.index if i.startswith('w')]
    indf.drop('rch_1', inplace=True)
    svs = np.max([int(i.split('_')[-1]) for i in indf.columns if 'right_sing' in i.lower()])

    if(indf['ident'].sum() == svs):
        indf.sort_values(by='right_sing_vec_1', ascending=False, inplace=True)
    else:
        indf.sort_values(by='ident', ascending=False, inplace=True)
    dfplot = indf.drop('ident', axis=1)

    dfplot = dfplot.drop('rch_0')

    curr_cmap = 'nipy_spectral_r'

    fig = plt.figure(figsize=(12, 4))
    cax=plt.gca()
    dfplot.plot.bar(ax=cax, stacked=True, legend=False, cmap=curr_cmap)
    # setup the normalization and the colormap
    normalize = mcolors.Normalize(vmin=0, vmax=svs+1)
    # setup the colorbar
    #scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=curr_cmap)
    #scalarmappaple.set_array(svs)
    #cbar = plt.colorbar(scalarmappaple)
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])

    cb = mpl.colorbar.ColorbarBase(ax=ax2, cmap=curr_cmap, norm=normalize, ticks=np.linspace(0,svs,svs+1),
                                   boundaries=np.linspace(0,svs,svs+1),
                                   format='%1i')

    #cbar.ax.get_yaxis().set_ticks([int(i) for i in cbar.ax.get_yaxis().ticks])
    return indf
