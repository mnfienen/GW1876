import os
import numpy as np
import pandas as pd
import pyemu
import matplotlib.pyplot as plt
import freyberg_setup as frey_mod

EXE_DIR = os.path.join("..","..","bin")
WORKING_DIR = frey_mod.WORKING_DIR_KR
MODEL_NAM = "freyberg.nam"
PST_NAME = frey_mod.PST_NAME_KR
NUM_SLAVES = 20
NUM_STEPS_RESPSURF = 40

def run_respsurf(par_names=None, pstfile=None, WORKING_DIR=frey_mod.WORKING_DIR_KR):
    if pstfile is None:
        PST_NAME = frey_mod.PST_NAME_KR
    else:
        PST_NAME = pstfile
    pst = pyemu.Pst(os.path.join(WORKING_DIR,PST_NAME))
    par = pst.parameter_data
    pst.pestpp_options['sweep_parameter_csv_file'] = PST_NAME.replace('.pst', "sweep_in.csv")
    pst.pestpp_options['sweep_output_csv_file'] = PST_NAME.replace('.pst', "sweep_out.csv")
    pst.write(os.path.join(WORKING_DIR,PST_NAME))
    icount = 0
    if par_names is None:
        parnme1 = par.parnme[0]
        parnme2 = par.parnme[1]
    else:
        parnme1 = par_names[0]
        parnme2 = par_names[1]
    p1 = np.linspace(par.loc[parnme1,"parlbnd"],par.loc[parnme1,"parubnd"],NUM_STEPS_RESPSURF).tolist()
    p2 = np.linspace(par.loc[parnme2,"parlbnd"],par.loc[parnme2,"parubnd"],NUM_STEPS_RESPSURF).tolist()
    p1_vals,p2_vals = [],[]
    for p in p1:
        p1_vals.extend(list(np.zeros(NUM_STEPS_RESPSURF)+p))
        p2_vals.extend(p2)
    df = pd.DataFrame({parnme1:p1_vals,parnme2:p2_vals})
    for cp in par.parnme.values:
        if cp not in df.columns:
            df[cp] = par.loc[cp].parval1
    df.to_csv(os.path.join(WORKING_DIR,PST_NAME.replace('.pst',"sweep_in.csv")))

    os.chdir(WORKING_DIR)
    pyemu.helpers.start_slaves('.', 'sweep', PST_NAME, num_slaves=NUM_SLAVES, master_dir='.')
    os.chdir("..")

def plot_response_surface(parnames, pstfile, WORKING_DIR=frey_mod.WORKING_DIR_KR,
                          nanthresh=None,alpha=0.5, label=True, maxresp=None):
    p1,p2 = parnames
    df_in = pd.read_csv(os.path.join(WORKING_DIR, pstfile.replace('.pst',"sweep_in.csv")))
    df_out = pd.read_csv(os.path.join(WORKING_DIR, pstfile.replace('.pst',"sweep_out.csv")))
    resp_surf = np.zeros((NUM_STEPS_RESPSURF, NUM_STEPS_RESPSURF))
    p1_values = df_in[p1].unique()
    p2_values = df_in[p2].unique()
    c = 0
    for i, v1 in enumerate(p1_values):
        for j, v2 in enumerate(p2_values):
            resp_surf[j, i] = df_out.loc[c, "phi"]
            c += 1
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    X, Y = np.meshgrid(p1_values, p2_values)
    if nanthresh is not None:
        resp_surf = np.ma.masked_where(resp_surf > nanthresh, resp_surf)
    p = ax.pcolor(X, Y, resp_surf, alpha=alpha, cmap="nipy_spectral")
    plt.colorbar(p)
    if maxresp is None:
        maxresp = np.max(resp_surf)
    c = ax.contour(X, Y, resp_surf,
                   levels=np.array([0.001, 0.01, 0.02, 0.05, .1, .2, .5])*maxresp,
                   colors='k')
    plt.title('min $\Phi$ = {0:.2f}'.format(np.nanmin(resp_surf)))
    if label:
        plt.clabel(c)
    ax.set_xlim(p1_values.min(), p1_values.max())
    ax.set_ylim(p2_values.min(), p2_values.max())
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    return fig, ax

def add_trajectory_to_plot(fig,ax,casename, title, working_dir, pst_name):
    obfun = pd.read_csv(os.path.join(working_dir,pst_name.replace('.pst','.{}.iobj'.format(casename))))
    pars=pd.read_csv(os.path.join(working_dir,pst_name.replace('.pst','.{}.ipar'.format(casename))))
    ax.plot(pars.hk1.values,pars.cal_flux.values, 'kx-')
    ax.set_title(title)
    plt.savefig('response_surface_{}.png'.format(casename))
    return pars, obfun