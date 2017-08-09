import os
import sys
import shutil
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import pyemu

PREFIX = "pp"
EXE_DIR = os.path.join("..","bin")
WORKING_DIR = 'freyberg_' + PREFIX
BASE_MODEL_DIR = os.path.join("..","models","Freyberg","Freyberg_Truth")
BASE_MODEL_NAM = "freyberg.truth.nam"
MODEL_NAM = "freyberg.nam"
#PST_NAME = WORKING_DIR+".pst"
PST_NAME = "freyberg.pst"
NUM_SLAVES = 10
NUM_STEPS_RESPSURF = 50
EXE_PREF = ''
if "window" not in platform.platform().lower():
    EXE_PREF = "./"

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

    # hack for modpath crap
    mp_files = [f for f in os.listdir(BASE_MODEL_DIR) if ".mp" in f.lower()]
    for mp_file in mp_files:
        shutil.copy2(os.path.join(BASE_MODEL_DIR,mp_file),os.path.join(WORKING_DIR,mp_file))
    shutil.copy2(os.path.join(BASE_MODEL_DIR,"freyberg.locations"),os.path.join(WORKING_DIR,"freyberg.locations"))
    np.savetxt(os.path.join(WORKING_DIR,"ibound.ref"),m.bas6.ibound[0].array,fmt="%2d")
    os.chdir(WORKING_DIR)
    pyemu.helpers.run("mp6 {0}".format(MODEL_NAM.replace('.nam','.mpsim')))
    os.chdir("..")
    ept_file = os.path.join(WORKING_DIR,MODEL_NAM.replace(".nam",".mpenpt"))
    shutil.copy2(ept_file,ept_file+".truth")
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


def setup_pest():

    os.chdir(WORKING_DIR)

    # setup hk pilot point parameters
    m = flopy.modflow.Modflow.load(MODEL_NAM,check=False)
    pp_space = 4
    df_pp = pyemu.gw_utils.setup_pilotpoints_grid(ml=m,prefix_dict={0:["hk"]},
                                          every_n_cell=pp_space)
    pp_file = "hkpp.dat"

    # setup hyd mod
    pyemu.gw_utils.modflow_hydmod_to_instruction_file(MODEL_NAM.replace(".nam",".hyd.bin"))
    pyemu.gw_utils.modflow_read_hydmod_file(MODEL_NAM.replace(".nam",".hyd.bin.truth"))
    df_hyd = pd.read_csv(MODEL_NAM.replace(".nam",".hyd.bin.truth.dat"),delim_whitespace=True)
    df_hyd.index = df_hyd.obsnme

    # setup list file water budget obs
    shutil.copy2(m.name+".list.truth",m.name+".list")
    df_wb = pyemu.gw_utils.setup_mflist_budget_obs(m.name+".list")

    # setup potential water budget obs
    # perlen = pd.to_datetime(m._start_datetime) + pd.to_timedelta(np.cumsum(m.dis.perlen.array),unit='d')
    # #obsnme = []
    # print(perlen,m.dis.perlen.array)
    # with open("freyberg.hds.dat.ins",'w') as f:
    #     f.write("pif ~\n")
    #     for dt in perlen:
    #         dt_str = dt.strftime('%Y%m%d')
    #         for i in range(m.nrow):
    #             for j in range(m.ncol):
    #                 oname = "i{0:02d}j{1:02d}_{2}".format(i,j,dt_str)
    #                 f.write("l1 !{0}!\n".format(oname))
    #             #obsnme.append(oname)
    # hds = flopy.utils.HeadFile("freyberg.hds")
    # f = open('freyberg.hds.dat','wb')
    # for data in hds.get_alldata():
    #     data = data.flatten()
    #     np.savetxt(f,data,fmt='%15.6E')
    # f.close()
    # pyemu.helpers.run("inschek freyberg.hds.dat.ins freyberg.hds.dat")
    # df_po = pd.read_csv("freyberg.hds.dat.obf",delim_whitespace=True,names=["obsnme","obsval"])
    # df_po.index = df_po.obsnme

    # set particle travel time obs
    endpoint_file = 'freyberg.mpenpt'
    lines = open(endpoint_file, 'r').readlines()
    items = lines[-1].strip().split()
    travel_time = float(items[4]) - float(items[3])
    with open("freyberg.travel.ins",'w') as f:
        f.write("pif ~\n")
        f.write("l1 w !travel_time!\n")
    with open("freyberg.travel","w") as f:
        f.write("travel_time {0}\n".format(travel_time))

    forecast_names = list(df_wb.loc[df_wb.obsnme.apply(lambda x: "riv" in x and "flx" in x),"obsnme"])
    forecast_names.append("travel_time")

    # setup wel parameters - history and future
    wel_fmt = {"l":lambda x: '{0:10.0f}'.format(x)}
    wel_fmt["r"] = wel_fmt['l']
    wel_fmt["c"] = wel_fmt['l']
    wel_fmt["flux"] = lambda x: '{0:15.6E}'.format(x)

    wel_files = ["WEL_0001.dat","WEL_0002.dat"]
    w_dfs = []
    for iwel,wel_file in enumerate(wel_files):

        df_wel = pd.read_csv(wel_file,delim_whitespace=True,header=None,names=["l","r","c","flux"])
        df_wel.loc[:,"parnme"] = df_wel.apply(lambda x: "w{0}_r{1:02.0f}_c{2:02.0f}".format(iwel,x.r,x.c),axis=1)
        df_wel.loc[:,"tpl_str"] = df_wel.parnme.apply(lambda x: "~  {0}   ~".format(x))
        f_tpl = open(wel_file+".temp.tpl",'w')
        f_tpl.write('ptf ~\n')
        f_tpl.write("        "+df_wel.loc[:,["l","r","c","tpl_str"]].to_string(index=False,header=False,formatters=wel_fmt))
        f_tpl.close()
        w_dfs.append(df_wel.loc[:,["parnme","tpl_str"]])
        df_wel.loc[:,"flux"] = 1.0
        with open(wel_file+".temp",'w') as f:
            f.write(df_wel.loc[:,["l","r","c","flux"]].to_string(index=False,header=False,formatters=wel_fmt))
        shutil.copy2(wel_file,wel_file+".bak")
    df_wel = pd.concat(w_dfs)
    df_wel.index = df_wel.parnme

    # setup rch parameters - history and future
    f_in = open(MODEL_NAM.replace(".nam",".rch"),'r')
    f_tpl = open(MODEL_NAM.replace(".nam",".rch.tpl"),'w')
    f_tpl.write("ptf ~\n")
    r_count = 0
    for line in f_in:
        raw = line.strip().split()
        if "open" in line.lower() and r_count < 2:
            raw[2] = "~  rch_{0}   ~".format(r_count)
            r_count += 1
        line = ' '.join(raw)
        f_tpl.write(line+'\n')
    f_in.close()
    f_tpl.close()

    # build lists of tpl, in, ins, and out files
    tpl_files = [f for f in os.listdir(".") if f.endswith(".tpl")]
    in_files = [f.replace(".tpl","") for f in tpl_files]

    ins_files = [f for f in os.listdir(".") if f.endswith(".ins")]
    out_files = [f.replace(".ins","") for f in ins_files]

    # build a control file
    pst = pyemu.pst_utils.pst_from_io_files(tpl_files,in_files,
                                            ins_files,out_files)

    # set some observation attribues
    obs = pst.observation_data
    obs.loc[df_wb.obsnme,"obgnme"] = df_wb.obgnme
    obs.loc[df_wb.obsnme,"weight"] = 0.0
    obs.loc[forecast_names,"weight"] = 0.0
    #obs.loc[df_po.obsnme,"obsval"] = df_po.obsval
    #obs.loc[df_po.obsnme,"weight"] = 0.0
    obs.loc[df_hyd.obsnme,"obsval"] = df_hyd.obsval
    c_names = df_hyd.loc[df_hyd.obsnme.apply(lambda x: x.startswith("cr") and "19700102" in x),"obsnme"]
    noise = np.random.normal(0.0,2.0,c_names.shape[0])
    obs.loc[df_hyd.obsnme,"weight"] = 0.0
    obs.loc[df_hyd.obsnme,"obsval"] = df_hyd.obsval
    obs.loc[c_names,"obsval"] += noise
    obs.loc[c_names,"weight"] = 5.0
    og_dict = {'c':"cal_wl","f":"fore_wl","p":"pot_wl"}
    obs.loc[df_hyd.obsnme,"obgnme"] = df_hyd.obsnme.apply(lambda x: og_dict[x.split('_')[0][0]])
    #obs.loc["travel_time","obsval"] = travel_time
    # set some parameter attribs
    par = pst.parameter_data
    par.loc[:,"parval1"] = 1.0
    par.loc[:,"parubnd"] = 1.25
    par.loc[:,"parlbnd"] = 0.75
    par.loc[df_pp.parnme,"parval1"] = 5.0
    par.loc[df_pp.parnme,"parlbnd"] = 0.5
    par.loc[df_pp.parnme,"parubnd"] = 50.0
    par.loc[:,"pargp"] = par.parnme.apply(lambda x: x.split('_')[0])
    par.loc[df_pp.parnme,"pargp"] = "hk"

    pst.model_command = ["python forward_run.py"]
    pst.control_data.pestmode = "regularization"
    pst.pestpp_options["forecasts"] = ','.join(forecast_names)
    pst.pestpp_options["n_iter_base"] = -1
    pst.pestpp_options["n_iter_super"] = 3
    pst.control_data.noptmax = 0
    a = float(pp_space) * m.dis.delr.array[0] * 3.0
    v = pyemu.geostats.ExpVario(contribution=1.0,a=a)
    gs = pyemu.geostats.GeoStruct(variograms=[v],transform="log")
    ok = pyemu.geostats.OrdinaryKrige(gs,pyemu.gw_utils.pp_file_to_dataframe(pp_file))
    ok.calc_factors_grid(m.sr,var_filename="pp_var.ref")
    ok.to_grid_factors_file(pp_file+".fac")
    #plt.imshow(np.loadtxt("pp_var.ref"))
    #plt.savefig("pp_var.ref.png")

    # first order Tikhonov
    #cov = pyemu.helpers.pilotpoint_prior_builder(pst,{gs:[pp_file+".tpl"]},sigma_range=6)
    #pyemu.helpers.first_order_pearson_tikhonov(pst,cov)

    # zero order Tikhonov
    pyemu.helpers.zero_order_tikhonov(pst)

    pst.write(PST_NAME.replace(".pst",".init.pst"))

    with open("forward_run.py",'w') as f:
        f.write("import os\nimport shutil\nimport pandas as pd\nimport numpy as np\nimport pyemu\nimport flopy\n")
        f.write("wel_files = ['WEL_0001.dat','WEL_0002.dat']\n")
        f.write("names=['l','r','c','flux']\n")
        f.write("wel_fmt = {'l':lambda x: '{0:10.0f}'.format(x)}\n")
        f.write("wel_fmt['r'] = wel_fmt['l']\n")
        f.write("wel_fmt['c'] = wel_fmt['l']\n")
        f.write("wel_fmt['flux'] = lambda x: '{0:15.6E}'.format(x)\n")
        f.write("for wel_file in wel_files:\n")
        f.write("    df_w = pd.read_csv(wel_file+'.bak',header=None,names=names,delim_whitespace=True)\n")
        f.write("    df_t = pd.read_csv(wel_file+'.temp',header=None,names=names,delim_whitespace=True)\n")
        f.write("    df_t.loc[:,'flux'] *= df_w.flux\n")
        f.write("    with open(wel_file,'w') as f:\n")
        f.write("        f.write('        '+df_t.loc[:,names].to_string(index=None,header=None,formatters=wel_fmt)+'\\n')\n")
        f.write("shutil.copy2('WEL_0002.dat','WEL_0003.dat')\n")
        f.write("pyemu.gw_utils.fac2real('hkpp.dat',factors_file='hkpp.dat.fac',out_file='hk_layer_1.ref')\n")
        f.write("pyemu.helpers.run('mfnwt {0} >_mfnwt.stdout')\n".format(MODEL_NAM))
        f.write("pyemu.helpers.run('mp6 freyberg.mpsim >_mp6.stdout')\n")
        f.write("pyemu.gw_utils.apply_mflist_budget_obs('{0}')\n".format(MODEL_NAM.replace(".nam",".list")))
        f.write("hds = flopy.utils.HeadFile('freyberg.hds')\n")
        f.write("f = open('freyberg.hds.dat','wb')\n")
        f.write("for data in hds.get_alldata():\n")
        f.write("    data = data.flatten()\n")
        f.write("    np.savetxt(f,data,fmt='%15.6E')\n")
        f.write("endpoint_file = 'freyberg.mpenpt'\nlines = open(endpoint_file, 'r').readlines()\n")
        f.write("items = lines[-1].strip().split()\ntravel_time = float(items[4]) - float(items[3])\n")
        f.write("with open('freyberg.travel', 'w') as ofp:\n")
        f.write("    ofp.write('travetime {0:15.6e}{1}'.format(travel_time, '\\n'))\n")
        f.write("pyemu.gw_utils.modflow_read_hydmod_file('freyberg.hyd.bin')\n")

    #os.system("pestchek {0}".format(PST_NAME))
    pyemu.helpers.run("pestchek {0}".format(PST_NAME))
    pyemu.helpers.run("pestpp {0}".format(PST_NAME.replace(".pst",".init.pst")))
    pst.control_data.noptmax = 8
    pst.write(PST_NAME)
    os.chdir("..")


def run():
    pass


def run_pe(pst_name=None):
    if pst_name is None:
        pst_name = PST_NAME
    #os.system("pestpp {0} 1>_pestpp_stdout 2>_pestpp_stderr".format(PST_NAME))
    #pyemu.helpers.run("pestpp {0} 1>_pestpp_stdout 2>_pestpp_stderr".format(PST_NAME))
    os.chdir(WORKING_DIR)
    #pyemu.helpers.run("pestpp {0}".format(PST_NAME))
    pyemu.helpers.start_slaves('.','pestpp',pst_name,num_slaves=NUM_SLAVES,master_dir='.')
    pst = pyemu.Pst(PST_NAME)
    pst.parrep(PST_NAME.replace(".pst",".parb"))
    pst.control_data.noptmax = 0
    pst.write(PST_NAME.replace(".pst",".final.pst"))
    pyemu.helpers.run("pestpp {0}".format(pst_name.replace(".pst",".final.pst")))
    os.chdir("..")
    #m = flopy.modflow.Modflow.load(MODEL_NAM,model_ws=WORKING_DIR)
    #m.lpf.hk[0].plot(colorbar=True)


def run_fosm():
    jco = os.path.join(WORKING_DIR,PST_NAME.replace('.pst','.jcb'))
    assert os.path.exists(jco),"jco not found:{0}".format(jco)
    sc = pyemu.Schur(jco=jco)
    par_sum = sc.get_parameter_summary()
    par = sc.pst.parameter_data
    #par_sum.loc[par.parnme,"prior_expt"] = par.parval1
    #sc.pst.parrep(os.path.join(WORKING_DIR,PST_NAME.replace(".pst",".parb")))
    #par_sum.loc[par.parnme,"post_expt"] = par.parval1
    par_sum.to_csv(jco+'par.csv')

    fore_sum = sc.get_forecast_summary()
    fore_sum.to_csv(jco+".fore.csv")

    ev = pyemu.ErrVar(jco=jco)
    sing_vals = [1,int(ev.pst.nnz_obs/4),int(ev.pst.nnz_obs/2)]
    for sing_val in sing_vals:
        ident = ev.get_identifiability_dataframe(sing_val)
        ident.sort_values(by="ident",inplace=True,ascending=False)
        ident.to_csv(os.path.join(WORKING_DIR,PST_NAME.replace(".pst",".ident.{0}.csv").format(sing_val)))
        ident.ident.iloc[:15].plot(kind="bar")


def run_dataworth():
    jco = os.path.join(WORKING_DIR,PST_NAME.replace('.pst','.jcb'))
    assert os.path.exists(jco),"jco not found:{0}".format(jco)
    sc = pyemu.Schur(jco=jco)
    df_ao = sc.get_added_obs_importance(base_obslist=[],reset_zero_weight=True)
    df_ao.to_csv(jco+".addobs.csv")

    df_pc = sc.get_par_contribution()
    df_pc.to_csv(jco+".parcontrib.csv")


def run_mc():
    pst = pyemu.Pst(os.path.join(WORKING_DIR,PST_NAME))
    mc = pyemu.MonteCarlo(pst=pst)
    mc.draw(1000)
    mc.parensemble.to_csv(os.path.join(WORKING_DIR,"sweep_in.csv"))
    os.chdir(WORKING_DIR)
    #pyemu.helpers.run("sweep {0}".format(PST_NAME))
    pyemu.helpers.start_slaves('.','sweep',PST_NAME,num_slaves=NUM_SLAVES,master_dir='.')
    os.chdir("..")
    df_obs = pd.read_csv(os.path.join(WORKING_DIR,"sweep_out.csv"))

    # to some filtering?

    df_obs.loc[:,pst.pestpp_options["forecasts"].split(',')].to_csv(os.path.join(WORKING_DIR,PST_NAME+".mc.fore.csv"))
    df_obs.loc[:,pst.nnz_obs_names].to_csv(os.path.join(WORKING_DIR,PST_NAME+".mc.obs.csv"))

def run_gsa():
    os.chdir(WORKING_DIR)
    pyemu.helpers.start_slaves('.','gsa',PST_NAME,num_slaves=NUM_SLAVES,master_dir='.')
    os.chdir("..")
    df = pd.read_csv(os.path.join(WORKING_DIR,PST_NAME.replace(".pst",".msn")),skipinitialspace=True)
    print(df.columns)
    df.loc[:,"parnme"] = df.parameter_name.apply(lambda x: x.lower().replace("log(",''.replace(')','')))



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

def run_ies():
    pass


if __name__ == "__main__":
    setup_model()
    #setup_pest()
    #run_pe()
    #run_fosm()
    #run_dataworth()
    #run_respsurf()
    #run_gsa()