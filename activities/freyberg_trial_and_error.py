import os
import shutil
import platform
import numpy as np
import pandas as pd
import flopy
import pyemu
import matplotlib.pyplot as plt

PREFIX = "trialerror"
EXE_DIR = os.path.join("..","bin")
WORKING_DIR = 'freyberg_' + PREFIX
BASE_MODEL_DIR = os.path.join("..","models","Freyberg","Freyberg_Truth")
BASE_MODEL_NAM = "freyberg.truth.nam"
MODEL_NAM = "freyberg.nam"
PST_NAME = WORKING_DIR+".pst"
NUM_SLAVES = 10
NUM_STEPS_RESPSURF = 50

def setup_model():

    if "window" in platform.platform().lower():
        exe_files = [f for f in os.listdir(EXE_DIR) if f.endswith('exe')]
    else:
        exe_files = [f for f in os.listdir(EXE_DIR) if not f.endswith('exe')]
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)
    for exe_file in exe_files:
        shutil.copy2(os.path.join(EXE_DIR,exe_file),os.path.join(WORKING_DIR,exe_file))

    print(os.listdir(BASE_MODEL_DIR))
    m = flopy.modflow.Modflow.load(BASE_MODEL_NAM,model_ws=BASE_MODEL_DIR,check=False)
    m.version = "mfnwt"
    m.change_model_ws(WORKING_DIR)
    m.name = MODEL_NAM.split(".")[0]
    m.remove_package("PCG")
    flopy.modflow.ModflowUpw(m,hk=m.lpf.hk,vka=m.lpf.vka,
                             ss=m.lpf.ss,sy=m.lpf.ss,
                             laytyp=m.lpf.laytyp,ipakcb=53)
    m.remove_package("LPF")
    flopy.modflow.ModflowNwt(m)
    m.write_input()

    m.exe_name = os.path.abspath(os.path.join(m.model_ws,"mfnwt"))
    m.run_model()
    hyd_out = os.path.join(WORKING_DIR,MODEL_NAM.replace(".nam",".hyd.bin"))
    shutil.copy2(hyd_out,hyd_out+'.truth')
    list_file = os.path.join(WORKING_DIR,MODEL_NAM.replace(".nam",".list"))
    shutil.copy2(list_file,list_file+".truth")

    m.upw.hk = m.upw.hk.array.mean()
    m.upw.hk[0].format.free = True

    wel_data_sp1 = m.wel.stress_period_data[0]
    #wel_data_sp1["flux"] = np.ceil(wel_data_sp1["flux"],order=)
    wel_data_sp1["flux"] = [round(f,-2) for f in wel_data_sp1["flux"]]
    wel_data_sp2 = wel_data_sp1.copy()
    wel_data_sp2["flux"] *= 1.2

    r = np.round(m.rch.rech[0].array.mean(),5)
    m.rch.rech[0] = r
    m.rch.rech[0].format.free = True
    m.external_path = '.'
    #m.oc.chedfm = "(20f16.6)"
    #output_idx = m.output_fnames.index("freyberg.hds")
    #m.output_binflag[output_idx] = False
    m.write_input()

    m.exe_name = os.path.abspath(os.path.join(m.model_ws,"mfnwt"))
    m.run_model()

    # hack for modpath crap
    mp_files = [f for f in os.listdir(BASE_MODEL_DIR) if ".mp" in f.lower()]
    for mp_file in mp_files:
        shutil.copy2(os.path.join(BASE_MODEL_DIR,mp_file),os.path.join(WORKING_DIR,mp_file))
    shutil.copy2(os.path.join(BASE_MODEL_DIR,"freyberg.locations"),os.path.join(WORKING_DIR,"freyberg.locations"))
    np.savetxt(os.path.join(WORKING_DIR,"ibound.ref"),m.bas6.ibound[0].array,fmt="%2d")

    pyemu.helpers.run('mp6 freyberg.mpsim')
    shutil.copy2('freyberg.mpenpt','freyberg.mpenpt.truth')


def setup_pest():

    os.chdir(WORKING_DIR)

    m = flopy.modflow.Modflow.load(MODEL_NAM,check=False)

    df_wb = pyemu.gw_utils.setup_mflist_budget_obs(m.name+".list")

    df_junk = pyemu.gw_utils.modflow_hydmod_to_instruction_file(MODEL_NAM.replace('nam', 'hyd.bin'))

    df_hds, outfile = pyemu.gw_utils.modflow_read_hydmod_file(MODEL_NAM.replace('nam', 'hyd.bin.truth'))

    # setup rch parameters - only for calibration period
    with open(MODEL_NAM.replace(".nam",".rch"),'r') as f_in:
        with open(MODEL_NAM.replace(".nam",".rch.tpl"),'w') as f_tpl:
            f_tpl.write("ptf ~\n")
            r_count = 0
            for line in f_in:
                raw = line.strip().split()
                if "open" in line.lower() and r_count < 1:
                    raw[2] = "~  rch_{0}   ~".format(r_count)
                    r_count += 1
                line = ' '.join(raw)
                f_tpl.write(line+'\n')

    # set up HK1 parameter
    with open(MODEL_NAM.replace(".nam", ".upw"), 'r') as f_in:
        with open(MODEL_NAM.replace(".nam", ".upw.tpl"), 'w') as f_tpl:
            f_tpl.write("ptf ~\n")
            for line in f_in:
                raw = line.strip().split()
                if "open" in line.lower() and 'hk_layer_1.ref' in line.lower():
                    raw = ['CONSTANT', '~  hk1   ~']
                line = ' '.join(raw)
                f_tpl.write(line + '\n')

    endpoint_file = 'freyberg.mpenpt.truth'
    lines = open(endpoint_file, 'r').readlines()
    items = lines[-1].strip().split()
    travel_time = float(items[4]) - float(items[3])

    with open('freyberg.travel', 'w') as ofp:
        ofp.write('travetime {0:15.6e}{1}'.format(travel_time, '\n'))

    with open("freyberg.travel.ins",'w') as f:
        f.write("pif ~\n")
        f.write("l1 w !travel_time!\n")

    tpl_files = [f for f in os.listdir(".") if f.endswith(".tpl")]
    in_files = [f.replace(".tpl", "") for f in tpl_files]

    ins_files = [f for f in os.listdir(".") if f.endswith(".ins")]
    out_files = [f.replace(".ins", "") for f in ins_files]

    pst = pyemu.pst_utils.pst_from_io_files(tpl_files,in_files,
                                            ins_files,out_files)

    hk_start = m.upw.hk.array.mean()


    pars = pst.parameter_data
    pars.loc[pars.parnme == 'hk1', 'parval1'] = hk_start
    pars.loc[pars.parnme == 'hk1', 'parlbnd'] = hk_start / 10.0
    pars.loc[pars.parnme == 'hk1', 'parubnd'] = hk_start * 10.0

    pars.loc[pars.parnme == 'rch_0', 'parval1'] = 1.0
    pars.loc[pars.parnme == 'rch_0', 'parlbnd'] = 0.5
    pars.loc[pars.parnme == 'rch_0', 'parubnd'] = 1.5


    obs = pst.observation_data
    obs.loc[df_wb.obsnme,"obgnme"] = df_wb.obgnme
    obs.loc[df_wb.obsnme,"weight"] = 0.0
    obs.loc[obs.obsnme == 'flx_river_l_19700102', 'weight'] = 0.01
    obs.loc[df_hds.obsnme,"obgnme"] = 'head'
    obs.loc[df_hds.obsnme,"weight"] = [1.0 if i.startswith("c") and i.endswith('19700102')
                                       else 0.0 for i in df_hds.obsnme]

    obs.loc[df_hds.obsnme, "obgnme"] = ['forecasthead' if i.startswith("f") and
                                                          i.endswith('19750102') else
                                        'head' for i in df_hds.obsnme]


    obs['obgnme'] = ['calhead' if i.startswith("c") and j == 1 else k for i,j,k in zip(obs.obsnme,
                                                                                       obs.weight,
                                                                                       obs.obgnme)]

    obs.loc['flx_river_l_19750102', 'ognme'] = 'forecastflux'
    obs.loc['travel_time', 'obgnme'] = 'forecasttravel'
    obs.loc["travel_time", "obsval"] = travel_time

    forecast_names = [i for i in df_hds.obsnme if i.startswith('f') and i.endswith('19750102')]
    forecast_names.append('flx_river_l_19750102')
    pst.pestpp_options["forecasts"] = ','.join(forecast_names)
    pst.control_data.noptmax = 0
    pst.model_command = ["python forward_run.py"]
    pst.write(PST_NAME)

    with open("forward_run.py",'w') as f:
        f.write("import os\nimport numpy as np\nimport pyemu\nimport flopy\nimport os\n")
        f.write("for cf in ['freyberg.travel','freyberg.hyd.bin','freyberg.list']:\n")
        f.write("    try:\n        os.remove(cf)\n    except:\n        pass\n")
        f.write("pyemu.helpers.run('mfnwt {0} >_mfnwt.stdout')\n".format(MODEL_NAM))
        f.write("pyemu.helpers.run('mp6 freyberg.mpsim >_mp6.stdout')\n")
        f.write("pyemu.gw_utils.apply_mflist_budget_obs('{0}')\n".format(MODEL_NAM.replace(".nam",".list")))
        f.write("pyemu.gw_utils.modflow_read_hydmod_file('{0}')\n".format(MODEL_NAM.replace(".nam",".hyd.bin")))
        f.write("endpoint_file = 'freyberg.mpenpt'\nlines = open(endpoint_file, 'r').readlines()\n")
        f.write("items = lines[-1].strip().split()\ntravel_time = float(items[4]) - float(items[3])\n")
        f.write("with open('freyberg.travel', 'w') as ofp:\n")
        f.write("    ofp.write('travetime {0:15.6e}{1}'.format(travel_time, '\\n'))\n")
    pyemu.helpers.run("pestchek {0}".format(PST_NAME))
    os.chdir("..")

def run():
    pass


def run_pe():
    #os.system("pestpp {0} 1>_pestpp_stdout 2>_pestpp_stderr".format(PST_NAME))
    #pyemu.helpers.run("pestpp {0} 1>_pestpp_stdout 2>_pestpp_stderr".format(PST_NAME))
    os.chdir(WORKING_DIR)
    pyemu.helpers.run("pestpp {0}".format(PST_NAME))
    os.chdir("..")

def run_fosm():
    jco = os.path.join(WORKING_DIR,PST_NAME.replace('.pst','.jcb'))
    assert os.path.exists(jco),"jco not found:{0}".format(jco)
    sc = pyemu.Schur(jco=jco)
    par_sum = sc.get_parameter_summary()
    par_sum.to_csv(jco+'par.csv')
    fore_sum = sc.get_forecast_summary()
    fore_sum.to_csv(jco+".fore.csv")

def run_dataworth():
    jco = os.path.join(WORKING_DIR,PST_NAME.replace('.pst','.jcb'))
    assert os.path.exists(jco),"jco not found:{0}".format(jco)
    sc = pyemu.Schur(jco=jco)
    df = sc.get_added_obs_importance(base_obslist=[],reset_zero_weight=True)
    df.to_csv(jco+".addobs.csv")


def run_mc():
    pst = pyemu.Pst(os.path.join(WORKING_DIR,PST_NAME))
    mc = pyemu.MonteCarlo(pst=pst)
    mc.draw(1000)
    mc.parensemble.to_csv(os.path.join(WORKING_DIR,"sweep_in.csv"))
    os.chdir(WORKING_DIR)
    pyemu.helpers.run("sweep {0}".format(PST_NAME))
    os.chdir("..")
    df_obs = pd.read_csv(os.path.join(WORKING_DIR,"sweep_out.csv"))

    # to some filtering?

    df_obs.loc[:,pst.pestpp_options["forecasts"].split(',')].to_csv(os.path.join(WORKING_DIR,PST_NAME+".mc.fore.csv"))
    df_obs.loc[:,pst.nnz_obs_names].to_csv(os.path.join(WORKING_DIR,PST_NAME+".mc.obs.csv"))

def run_gsa():
    with open(PST_NAME.replace(".pst",".gsa"),'w') as f:
        f.write("TODO!")
    os.chdir(WORKING_DIR)
    pyemu.helpers.run("gsa {0}".format(PST_NAME))
    os.chdir("..")

def run_respsurf(par_names=None):
    pst = pyemu.Pst(os.path.join(WORKING_DIR,PST_NAME))
    par = pst.parameter_data
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
    df.to_csv(os.path.join(WORKING_DIR,"sweep_in.csv"))

    os.chdir(WORKING_DIR)
    pyemu.helpers.start_slaves('.', 'sweep', PST_NAME, num_slaves=NUM_SLAVES, master_dir='.')
    os.chdir("..")

def plot_response_surface():
    df_in = pd.read_csv(os.path.join(WORKING_DIR, "sweep_in.csv"))
    df_out = pd.read_csv(os.path.join(WORKING_DIR, "sweep_out.csv"))
    resp_surf = np.zeros((NUM_STEPS_RESPSURF, NUM_STEPS_RESPSURF))

    c = 0
    for i, v1 in enumerate(df_in.hk1.values):
        for j, v2 in enumerate(df_in.rch_0.values):
            resp_surf[j, i] = df_out.loc[c, "phi"]
            c += 1
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    X, Y = np.meshgrid(hk_values, fx_values)
    #resp_surf = np.ma.masked_where(resp_surf > 5, resp_surf)
    p = ax.pcolor(X, Y, resp_surf, alpha=0.5, cmap="spectral")
    plt.colorbar(p)
    c = ax.contour(X, Y, resp_surf, levels=[0.1, 0.2, 0.5, 1, 2, 5], colors='k')
    plt.clabel(c)
    ax.set_xlim(hk_values.min(), hk_values.max())
    ax.set_ylim(rch_values.min(), frch_values.max())
    ax.set_xlabel("hk1 ($\\frac{L}{T}$)")
    ax.set_ylabel("rch ($L$)")

def rerun_new_pars(hk1=5.5, rch_0 = 1.0):
    pst = pyemu.Pst(os.path.join(WORKING_DIR,PST_NAME))
    pst.control_data.noptmax = 0
    pars = pst.parameter_data
    pars.loc[pars.parnme == 'hk1', 'parval1']   = hk1
    pars.loc[pars.parnme == 'rch_0', 'parval1'] = rch_0
    pst.write(os.path.join(WORKING_DIR,'onerun.pst'))
    os.chdir(WORKING_DIR)
    pyemu.helpers.run('pestpp onerun.pst')
    os.chdir('..')

    newpst = pyemu.Pst(os.path.join(WORKING_DIR,'onerun.pst'))
    res = newpst.res

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121)
    cal = res.loc[res.group == 'calhead']
    plt.plot(cal.measured, cal.modelled, '.')
    minmin = np.min([cal.measured.min(),cal.modelled.min()])
    maxmax = np.max([cal.measured.max(), cal.modelled.max()])
    plt.plot([minmin, maxmax],[minmin,maxmax], 'r')
    plt.xlabel('measured')
    plt.ylabel('modeled')
    plt.title('Calibration')

    ax = fig.add_subplot(122)
    fore = res.loc[res.group == 'forecasthead']
    plt.plot(fore.measured, fore.modelled, '.')
    minmin = np.min([fore.measured.min(), fore.modelled.min()])
    maxmax = np.max([fore.measured.max(), fore.modelled.max()])
    plt.plot([minmin, maxmax],[minmin,maxmax], 'r')
    plt.xlabel('measured')
    plt.ylabel('modeled')
    plt.title('Forecast')

    plt.show()


def run_ies():
    pass


if __name__ == "__main__":
    #setup_model()
    #setup_pest()
    #run_pe()
    run_fosm()
    run_dataworth()
