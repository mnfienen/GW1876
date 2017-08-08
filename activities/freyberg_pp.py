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
PST_NAME = WORKING_DIR+".pst"


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
    m.change_model_ws(WORKING_DIR)
    m.name = MODEL_NAM.split(".")[0]
    m.lpf.hk = m.lpf.hk.array.mean()
    m.lpf.hk[0].format.free = True
    wel_data_sp1 = m.wel.stress_period_data[0]
    #wel_data_sp1["flux"] = np.ceil(wel_data_sp1["flux"],order=)
    wel_data_sp1["flux"] = [round(f,-2) for f in wel_data_sp1["flux"]]
    wel_data_sp2 = wel_data_sp1.copy()
    wel_data_sp2["flux"] *= 1.2

    r = np.round(m.rch.rech[0].array.mean(),5)
    m.rch.rech[0] = r
    m.external_path = '.'
    #m.oc.chedfm = "(20f16.6)"
    #output_idx = m.output_fnames.index("freyberg.hds")
    #m.output_binflag[output_idx] = False
    m.write_input()

    m.exe_name = "mf2005"
    m.run_model()

    # hack for modpath crap
    mp_files = [f for f in os.listdir(BASE_MODEL_DIR) if ".mp" in f.lower()]
    for mp_file in mp_files:
        shutil.copy2(os.path.join(BASE_MODEL_DIR,mp_file),os.path.join(WORKING_DIR,mp_file))
    shutil.copy2(os.path.join(BASE_MODEL_DIR,"freyberg.locations"),os.path.join(WORKING_DIR,"freyberg.locations"))
    np.savetxt(os.path.join(WORKING_DIR,"ibound.ref"),m.bas6.ibound[0].array,fmt="%2d")

def setup_pest():

    os.chdir(WORKING_DIR)
    m = flopy.modflow.Modflow.load(MODEL_NAM,check=False)
    pp_space = 4
    df_pp = pyemu.gw_utils.setup_pilotpoints_grid(ml=m,prefix_dict={0:["hk"]},
                                          every_n_cell=pp_space)
    pp_file = "hkpp.dat"
    df_wb = pyemu.gw_utils.setup_mflist_budget_obs(m.name+".list")
    perlen = pd.to_datetime(m._start_datetime) + pd.to_timedelta(np.cumsum(m.dis.perlen.array),unit='d')
    #obsnme = []
    print(perlen,m.dis.perlen.array)
    with open("freyberg.hds.dat.ins",'w') as f:
        f.write("pif ~\n")
        for dt in perlen:
            dt_str = dt.strftime('%Y%m%d')
            for i in range(m.nrow):
                for j in range(m.ncol):
                    oname = "i{0:02d}j{1:02d}_{2}".format(i,j,dt_str)
                    f.write("l1 !{0}!\n".format(oname))
                #obsnme.append(oname)
    hds = flopy.utils.HeadFile("freyberg.hds")
    f = open('freyberg.hds.dat','wb')
    for data in hds.get_alldata():
        data = data.flatten()
        np.savetxt(f,data,fmt='%15.6E')
    f.close()
    pyemu.helpers.run("inschek freyberg.hds.dat.ins freyberg.hds.dat")
    df_po = pd.read_csv("freyberg.hds.dat.obf",delim_whitespace=True,names=["obsnme","obsval"])
    df_po.index = df_po.obsnme

    with open("freyberg.travel.ins",'w') as f:
        f.write("pif ~\n")
        f.write("l1 w !travel_time!\n")

    forecast_names = list(df_wb.loc[df_wb.obsnme.apply(lambda x: "riv" in x and "flx" in x),"obsnme"])
    forecast_names.append("travel_time")

    tpl_files = [f for f in os.listdir(".") if f.endswith(".tpl")]
    in_files = [f.replace(".tpl","") for f in tpl_files]

    ins_files = [f for f in os.listdir(".") if f.endswith(".ins")]
    out_files = [f.replace(".ins","") for f in ins_files]

    pst = pyemu.pst_utils.pst_from_io_files(tpl_files,in_files,
                                            ins_files,out_files)
    obs = pst.observation_data
    obs.loc[df_wb.obsnme,"obgnme"] = df_wb.obgnme
    obs.loc[df_wb.obsnme,"weight"] = 0.0
    obs.loc[forecast_names,"weight"] = 0.0
    obs.loc[df_po.obsnme,"obsval"] = df_po.obsval
    obs.loc[df_po.obsnme,"weight"] = 0.0
    pst.model_command = ["python forward_run.py"]
    pst.pestpp_options["forecasts"] = ','.join(forecast_names)
    pst.write(PST_NAME)

    a = float(pp_space) * m.dis.delr.array[0] * 3.0
    v = pyemu.geostats.ExpVario(contribution=1.0,a=a)
    gs = pyemu.geostats.GeoStruct(variograms=[v],transform="log")
    pp_files = [f for f in os.listdir(".") if f.endswith("pp.dat")]
    #for pp_file in pp_files:
    #    ok = pyemu.geostats.OrdinaryKrige(gs,pyemu.gw_utils.pp_file_to_dataframe(pp_file))
    ok = pyemu.geostats.OrdinaryKrige(gs,pyemu.gw_utils.pp_file_to_dataframe(pp_file))
    ok.calc_factors_grid(m.sr,var_filename="pp_var.ref")
    ok.to_grid_factors_file(pp_file+".fac")
    plt.imshow(np.loadtxt("pp_var.ref"))
    plt.savefig("pp_var.ref.png")

    fname = os.path.split(__file__)[-1].split(".")[0]
    with open("forward_run.py",'w') as f:
        #f.write("import sys\nsys.path.append('..')\n")
        #f.write("import {0}\n".format(fname))
        #f.write("{0}.forward_run()\n".format(fname))
        f.write("import os\nimport numpy as np\nimport pyemu\nimport flopy\n")
        #f.write("pp_files = [f for f in os.listdir('.') if f.endswith('pp.dat')]\n")
        #f.write("for pp_file in pp_files:\n")
        #f.write("    pyemu.gw_utils.fac2real(pp_file,factors_file=pp_file+'.fac',out_file='hk_layer_1.ref')\n")
        f.write("pyemu.gw_utils.fac2real('hkpp.dat',factors_file='hkpp.dat.fac',out_file='hk_layer_1.ref')\n")

        f.write("pyemu.helpers.run('mf2005 {0} >_mf2005.stdout')\n".format(MODEL_NAM))
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

    #os.system("pestchek {0}".format(PST_NAME))
    pyemu.helpers.run("pestchek {0}".format(PST_NAME))
    os.chdir("..")


#def forward_run():
    # pp_files = [f for f in os.listdir('.') if f.endswith("pp.dat")]
    # for pp_file in pp_files:
    #     pyemu.gw_utils.fac2real(pp_file,factors_file=pp_file+".fac")
    # os.system("mf2005 {0} >_mf2005.stdout".format(MODEL_NAM))
    # pyemu.gw_utils.apply_mflist_budget_obs(os.path.join(WORKING_DIR+".list"))


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

def run_respsurf():
    pass

def run_ies():
    pass


if __name__ == "__main__":
    #setup_model()
    #setup_pest()
    #run_pe()
    run_fosm()
    run_dataworth()
