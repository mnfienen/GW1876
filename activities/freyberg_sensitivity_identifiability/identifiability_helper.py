import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import pandas as pd
import numpy as np

def plot_id_bars(ev, numsing):


    indf = ev.get_identifiability_dataframe(singular_value=numsing)

    #[indf.drop(i, inplace=True) for i in indf.index if i.startswith('w')]
    indf.drop('rch_1', inplace=True)
    svs = np.max([int(i.split('_')[-1]) for i in indf.columns if 'right_sing' in i.lower()])

    indf.sort_values(by='ident', ascending=False, inplace=True)

    if np.squeeze(np.unique(indf.ident.unique()>0.99999999)):
        indf.sort_values(by='right_sing_vec_1', ascending=False, inplace=True)
    dfplot = indf.drop('ident', axis=1)

    curr_cmap = 'jet_r'

    plt.figure(figsize=(12, 8))
    axmain = plt.subplot2grid((1, 15), (0, 0), colspan=13)

    dfplot.plot.bar(ax = axmain, legend=None, stacked = True, width = 0.8, colormap = curr_cmap)
    axmain.set_ylabel('Identifiability')
    axmain.tick_params(axis='x')

    ax1 = plt.subplot2grid((1, 15), (0, 13))
    norm = mpl.colors.Normalize(vmin=1, vmax=svs)
    cb_bounds = np.linspace(0, svs, svs + 1).astype(int)[1:]
    cb_axis = np.arange(0, svs + 1, int(np.max([1, svs / 10.0])))
    cb_axis[0] = 1
    cb = mpl.colorbar.ColorbarBase(ax1, cmap=curr_cmap, norm=norm, boundaries=cb_bounds, orientation='vertical')
    cb.set_ticks(cb_axis)
    cb.set_ticklabels(cb_axis)
    cb.set_label('Number of singular values considered')

    return indf

def plot_identifiability_spatial(ev, nsingular, parlox, obslox, makelabels=False,figsize=(5,8)):
    ident_df = ev.get_identifiability_dataframe(nsingular)
    ident_df=ident_df[['ident']].join(parlox)
    [ident_df.drop(i, inplace=True) for i in ident_df.index if i.startswith('w')]
    [ident_df.drop(i, inplace=True) for i in ident_df.index if i.startswith('rch')]

    plt.figure(figsize=figsize)
    plt.plot(obslox.X, obslox.Y,'x')
    plt.scatter(ident_df.X, ident_df.Y, s=np.abs(ident_df.ident.values)*150,
                c=ident_df.ident.values, cmap='viridis')
    if makelabels:
        for cn, cg in parlox.groupby('pp_num'):
            plt.gca().annotate(cn[-2:], xy=(cg.X,cg.Y), fontsize=9)
    plt.axis('equal')
    plt.colorbar()
    plt.title('Identifiability with {0} singular values'.format(nsingular))
    plt.xlim(0,5000)
    plt.ylim(0,10000)
    plt.axis('off')