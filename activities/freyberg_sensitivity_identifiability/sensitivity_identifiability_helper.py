import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
import re
import sys
sys.path.append('..')
import freyberg_setup as fs
working_dir = fs.WORKING_DIR_PP

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


def plot_Jacobian(jac, figsize=(7,4), cmap='viridis', logtrans=True):
    f = plt.figure(figsize=figsize)
    ax = plt.axes([0, 0.05, 0.9, 0.9])  # left, bottom, width, height
    if logtrans:
        jcdata = np.log(np.abs(jac.df()))
    else:
        jcdata = jac.df()
    im = ax.imshow(jcdata, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.xticks(range(len(jac.col_names)), jac.col_names, rotation=90)
    plt.yticks(range(len(jac.row_names)), jac.row_names)

    ax.grid(False)
    cax = plt.axes([0.95, 0.05, 0.05, 0.9])
    plt.colorbar(mappable=im, cax=cax)


def get_par_obs_lox():
    obslox = pd.read_csv(os.path.join(working_dir,'freyberg.hyd'), delim_whitespace=True, usecols = [4,5,6],
                         index_col=2, skiprows = 1, header=None, names=['X','Y','obsname'])
    obslox = obslox.drop([i for i in obslox.index if not i.startswith('cr')], axis=0)
    parlox = pd.read_csv(os.path.join(working_dir,'hkpp.dat.tpl'), delim_whitespace=True, usecols=[0,1,2],
                        index_col=0, skiprows=1, header=None, names=['parname','X','Y'])
    parlox['pp_num'] = [re.findall('\d+',i)[0] for i in parlox.index.values]
    parlox.index=['hk{}'.format(i[-2:]) for i in parlox.index]
    return parlox, obslox

def plot_identifiability_spatial(ev, nsingular, makelabels=False,figsize=(5,8)):
    parlox,obslox = get_par_obs_lox()
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

def plot_jacobian_spatial(jac,cobs, figsize=(4,7)):
    parlox,obslox = get_par_obs_lox()
    sens = jac.df().loc[cobs]
    sens.drop('rch_1', inplace=True)
    sens.drop('rch_0', inplace=True)
    [sens.drop(i, inplace=True) for i in sens.index if i.startswith('w')]

    fig=plt.figure(figsize=figsize)
    plt.plot(parlox.X,parlox.Y,'kd',markersize=.8)
    scalefactor=5
    if 'flux' not in cobs:
        coblox = obslox.loc[cobs]
        plt.plot(coblox.X,coblox.Y,'kx', markersize=10)
        scalefactor=1000
    plt.scatter(parlox.X,parlox.Y, s=np.abs(sens.values)*scalefactor, c=sens.values, cmap='viridis')
    plt.axis('equal')
    plt.colorbar()
    plt.title('Sensitivity for {0}'.format(cobs))
    plt.xlim(0,5000)
    plt.ylim(0,10000)
    plt.axis('off')
    return fig