import os
import sys
import shutil
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import pyemu

EXE_DIR = os.path.join("..","..","bin")
WORKING_DIR_PP = 'freyberg_pp'
WORKING_DIR_GR = "freyberg_gr"
WORKING_DIR_ZN = "freyberg_zn"
WORKING_DIR_KR = "freyberg_kr"
WORKING_DIR_UN = "freyberg_un"
BASE_MODEL_DIR = os.path.join("..","..","models","Freyberg","Freyberg_Truth")
BASE_MODEL_NAM = "freyberg.truth.nam"
MODEL_NAM = "freyberg.nam"
PST_NAME_PP = "freyberg_pp.pst"
PST_NAME_GR = "freyberg_gr.pst"
PST_NAME_ZN = "freyberg_zn.pst"
PST_NAME_KR = "freyberg_kr.pst"
PST_NAME_UN = "freyberg_un.pst"
NUM_SLAVES = 10
NUM_STEPS_RESPSURF = 25


def setup_model(working_dir):

    if "window" in platform.platform().lower():
        exe_files = [f for f in os.listdir(EXE_DIR) if f.endswith('exe')]
    else:
        exe_files = [f for f in os.listdir(EXE_DIR) if not f.endswith('exe')]
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.mkdir(working_dir)
    for exe_file in exe_files:
        shutil.copy2(os.path.join(EXE_DIR,exe_file),os.path.join(working_dir,exe_file))

    print(os.listdir(BASE_MODEL_DIR))
    m = flopy.modflow.Modflow.load(BASE_MODEL_NAM,model_ws=BASE_MODEL_DIR,check=False)
    m.version = "mfnwt"
    m.change_model_ws(working_dir)
    m.name = MODEL_NAM.split(".")[0]
    m.remove_package("PCG")
    flopy.modflow.ModflowUpw(m,hk=m.lpf.hk,vka=m.lpf.vka,
                             ss=m.lpf.ss,sy=m.lpf.ss,
                             laytyp=m.lpf.laytyp,ipakcb=53)
    m.remove_package("LPF")
    flopy.modflow.ModflowNwt(m)
    m.write_input()
    m.exe_name = os.path.abspath(os.path.join(m.model_ws,"mfnwt"))
    #repair_hyd(m)
    m.run_model()
    # hack for modpath crap
    mp_files = [f for f in os.listdir(BASE_MODEL_DIR) if ".mp" in f.lower()]
    for mp_file in mp_files:
        shutil.copy2(os.path.join(BASE_MODEL_DIR,mp_file),os.path.join(working_dir,mp_file))
    shutil.copy2(os.path.join(BASE_MODEL_DIR,"freyberg.locations"),
                 os.path.join(working_dir,"freyberg.locations"))
    np.savetxt(os.path.join(working_dir,"ibound.ref"),m.bas6.ibound[0].array,fmt="%2d")
    os.chdir(working_dir)
    pyemu.helpers.run("mp6 {0}".format(MODEL_NAM.replace('.nam','.mpsim')))
    os.chdir("..")
    ept_file = os.path.join(working_dir,MODEL_NAM.replace(".nam",".mpenpt"))
    shutil.copy2(ept_file,ept_file+".truth")
    hyd_out = os.path.join(working_dir,MODEL_NAM.replace(".nam",".hyd.bin"))
    shutil.copy2(hyd_out,hyd_out+'.truth')
    list_file = os.path.join(working_dir,MODEL_NAM.replace(".nam",".list"))
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

def repair_hyd(m):
    os.remove(os.path.join(m.model_ws,m.name+".hyd"))
    obs = pd.read_csv(os.path.join(BASE_MODEL_DIR,"obs_loc.csv"))
    obs.loc[:,"rc"] = obs.apply(lambda x: (x.row,x.col),axis=1)
    obs_rc = list(obs.rc)
    ib = m.bas6.ibound[0].array
    lines = []
    for i in range(m.nrow):
        for j in range(m.ncol):
            if ib[i,j] == 0:
                continue
            x,y = m.sr.xcentergrid[i,j],m.sr.ycentergrid[i,j]
            r,c = i+1,j+1
            if (r,c) in obs_rc:
                prefixes = ['c','f']
            else:
                prefixes = ['p']
            for prefix in prefixes:
                name = "{0}r{1:02d}c{2:02d}".\
                    format(prefix,r,c)
                line = "BAS HD I 1 {0:7.3f} {1:7.3f} {2}\n".\
                    format(x,y,name)
                lines.append(line)
    with open(os.path.join(m.model_ws,m.name+".hyd"),'w') as f:
        f.write("{0} {1} -1.0e+10\n".format(len(lines),536))
        for line in lines:
            f.write(line)

def _get_base_pst(m):

     # setup hyd mod
    pyemu.gw_utils.modflow_hydmod_to_instruction_file(MODEL_NAM.replace(".nam",".hyd.bin"))
    pyemu.gw_utils.modflow_read_hydmod_file(MODEL_NAM.replace(".nam",".hyd.bin.truth"))
    df_hyd = pd.read_csv(MODEL_NAM.replace(".nam",".hyd.bin.truth.dat"),delim_whitespace=True)
    df_hyd.index = df_hyd.obsnme

    # setup list file water budget obs
    shutil.copy2(m.name+".list.truth",m.name+".list")
    df_wb = pyemu.gw_utils.setup_mflist_budget_obs(m.name+".list")

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


    #forecast_names = list(df_wb.loc[df_wb.obsnme.apply(lambda x: "riv" in x and "flx" in x),"obsnme"])
    forecast_names = [i for i in df_hyd.obsnme if i.startswith('f') and i.endswith('19750102')]
    forecast_names.append('flx_river_l_19750102')
    forecast_names.append("travel_time")


    # build lists of tpl, in, ins, and out files
    tpl_files = [f for f in os.listdir(".") if f.endswith(".tpl")]
    in_files = [f.replace(".tpl","") for f in tpl_files]

    ins_files = [f for f in os.listdir(".") if f.endswith(".ins")]
    out_files = [f.replace(".ins","") for f in ins_files]

    # build a control file
    pst = pyemu.Pst.from_io_files(tpl_files,in_files,
                                            ins_files,out_files)

    par = pst.parameter_data
    par.loc[:,"parval1"] = 1.0
    par.loc[:,"parubnd"] = 2.0
    par.loc[:,"parlbnd"] = 0.5
    hk_names = par.loc[par.parnme.apply(lambda x: x.startswith("hk")),"parnme"]
    par.loc[hk_names,"parval1"] = 5.0
    par.loc[hk_names,"parlbnd"] = 0.5
    par.loc[hk_names,"parubnd"] = 50.0
    par.loc[:,"pargp"] = par.parnme.apply(lambda x: x.split('_')[0])


    # set some observation attribues
    obs = pst.observation_data

    obs.loc[df_wb.obsnme,"obgnme"] = df_wb.obgnme
    obs.loc[df_wb.obsnme,"weight"] = 0.0
    obs.loc[forecast_names,"weight"] = 0.0
    obs.loc[df_hyd.obsnme,"obsval"] = df_hyd.obsval
    c_names = df_hyd.loc[df_hyd.obsnme.apply(lambda x: x.startswith("cr") and "19700102" in x),"obsnme"]
    np.random.seed(pyemu.en.SEED)
    noise = np.random.normal(0.0,2.0,c_names.shape[0])
    obs.loc[df_hyd.obsnme,"weight"] = 0.0
    obs.loc[df_hyd.obsnme,"obsval"] = df_hyd.obsval
    obs.loc[c_names,"obsval"] += noise
    obs.loc[c_names,"weight"] = 5.0
    obs.loc[df_hyd.obsnme,"obgnme"] = 'head'
    obs.loc[df_hyd.obsnme, "obgnme"] = ['forehead' if i.startswith("f") and
                                                          i.endswith('19750101') else
                                        'pothead' if i.startswith('p') and
                                                        i.endswith('19700102') else
                                        'head' for i in df_hyd.obsnme]


    obs['obgnme'] = ['calhead' if i.startswith("c") and j > 0  else k for i,j,k in zip(obs.obsnme,
                                                                                       obs.weight,
                                                                                       obs.obgnme)]

    obs.loc['flx_river_l_19750101', 'obgnme'] = 'foreflux'
    obs.loc['travel_time', 'obgnme'] = 'foretrav'

    obs.loc[obs.obsnme == 'flx_river_l_19700102', 'weight'] = 0.1
    obs.loc[obs.obsnme == 'flx_river_l_19700102', 'obgnme'] = 'calflux'

    # obs.loc[df_wb.obsnme,"obgnme"] = df_wb.obgnme
    # obs.loc[df_wb.obsnme,"weight"] = 0.0
    # obs.loc[forecast_names,"weight"] = 0.0
    # c_names = df_hyd.loc[df_hyd.obsnme.apply(lambda x: x.startswith("cr") and "19700102" in x),"obsnme"]
    # np.random.seed(pyemu.en.SEED)
    # noise = np.random.normal(0.0,2.0,c_names.shape[0])
    # obs.loc[df_hyd.obsnme,"weight"] = 0.0
    # obs.loc[df_hyd.obsnme,"obsval"] = df_hyd.obsval
    # obs.loc[c_names,"obsval"] += noise
    # obs.loc[c_names,"weight"] = 5.0
    # og_dict = {'c':"cal_wl","f":"fore_wl","p":"pot_wl"}
    # obs.loc[df_hyd.obsnme,"obgnme"] = df_hyd.obsnme.apply(lambda x: og_dict[x.split('_')[0][0]])
    #obs.loc["travel_time","obsval"] = travel_time
    pst.pestpp_options["forecasts"] = ','.join(forecast_names)
    return pst


def setup_pest_un_bareass():
    setup_model(WORKING_DIR_UN)
    os.chdir(WORKING_DIR_UN)

    # setup hk parameters
    m = flopy.modflow.Modflow.load(MODEL_NAM,check=False)
    with open("hk_layer_1.ref.tpl",'w') as f:
        f.write("ptf ~\n")
        for i in range(m.nrow):
            for j in range(m.ncol):
                f.write(" ~     hk   ~")
            f.write("\n")


    pst = _get_base_pst(m)
    obs = pst.observation_data
    keep = list(obs.loc[obs.obgnme=="calhead","obsnme"])
    keep.append(obs.loc[obs.obgnme=="forehead","obsnme"].iloc[0])
    keep.append("travel_time")
    pst.observation_data = obs.loc[keep,:]
    pst.instruction_files = [f for f in pst.instruction_files if "hyd" in f or "travel" in f]
    pst.output_files = [f for f in pst.output_files if "hyd" in f or "travel" in f]
    pst.pestpp_options.pop("forecasts")

    lines = []
    hyd_name = "freyberg.hyd.bin.dat.ins"
    with open(hyd_name,'r') as f:
        [f.readline() for _ in range(2)]
        for line in f:
            n = line.strip().split()[-1].replace("!","").lower()
            if n in keep:
                lines.append(line)
            else:
                lines.append("l1 \n")
    with open(hyd_name,"w") as f:
        f.write("pif ~\n")
        f.write("l1\n")
        [f.write(line) for line in lines]

    # set some parameter attribs
    par = pst.parameter_data
    

    obs = pst.observation_data
    obs.loc[obs.obsnme.apply(lambda x: x.startswith("flx")),"weight"] = 0.0

    pst.model_command = ["python forward_run.py"]
    pst.control_data.noptmax = 0
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["upgrade_augment"] = "false"
    pst.pestpp_options["lambdas"] = "0.1,1.0,10.0"
    pst.write(PST_NAME_UN.replace(".pst",".init.pst"))

    with open("forward_run.py",'w') as f:
        f.write("import os\nimport shutil\nimport pandas as pd\nimport numpy as np\nimport pyemu\nimport flopy\n")
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
    pst.control_data.noptmax = 8
    pst.write(PST_NAME_UN)
    pyemu.helpers.run("pestchek {0}".format(PST_NAME_UN))
    pyemu.helpers.run("pestpp {0}".format(PST_NAME_UN.replace(".pst",".init.pst")))
    os.chdir("..")

def setup_pest_un():
    setup_model(WORKING_DIR_UN)
    os.chdir(WORKING_DIR_UN)

    # setup hk parameters
    m = flopy.modflow.Modflow.load(MODEL_NAM,check=False)
    with open("hk_layer_1.ref.tpl",'w') as f:
        f.write("ptf ~\n")
        for i in range(m.nrow):
            for j in range(m.ncol):
                f.write(" ~     hk   ~")
            f.write("\n")


    pst = _get_base_pst(m)

    
    obs = pst.observation_data
    obs.loc[obs.obsnme.apply(lambda x: x.startswith("flx")),"weight"] = 0.0

    pst.model_command = ["python forward_run.py"]
    pst.control_data.noptmax = 0
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["upgrade_augment"] = "false"
    pst.pestpp_options["lambdas"] = "0.1,1.0,10.0"
    pst.write(PST_NAME_UN.replace(".pst",".init.pst"))

    with open("forward_run.py",'w') as f:
        f.write("import os\nimport shutil\nimport pandas as pd\nimport numpy as np\nimport pyemu\nimport flopy\n")
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
    pst.control_data.noptmax = 8
    pst.write(PST_NAME_UN)
    pyemu.helpers.run("pestchek {0}".format(PST_NAME_UN))
    pyemu.helpers.run("pestpp {0}".format(PST_NAME_UN.replace(".pst",".init.pst")))
    os.chdir("..")



def setup_pest_kr():
    setup_model(WORKING_DIR_KR)
    os.chdir(WORKING_DIR_KR)

    # setup hk parameters
    m = flopy.modflow.Modflow.load(MODEL_NAM,check=False)
    with open("hk_layer_1.ref.tpl",'w') as f:
        f.write("ptf ~\n")
        for i in range(m.nrow):
            for j in range(m.ncol):
                f.write(" ~     hk   ~")
            f.write("\n")

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

    pst = _get_base_pst(m)

    par = pst.parameter_data
    par.loc[["rch_0","rch_1"],"partrans"] = "fixed"

    obs = pst.observation_data
    obs.loc[obs.obsnme.apply(lambda x: x.startswith("flx")),"weight"] = 0.0

    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["upgrade_augment"] = "false"
    pst.pestpp_options["lambdas"] = "0.1,1.0,10.0"
    
    pst.model_command = ["python forward_run.py"]
    pst.control_data.noptmax = 0

    pst.write(PST_NAME_KR.replace(".pst",".init.pst"))

    with open("forward_run.py",'w') as f:
        f.write("import os\nimport shutil\nimport pandas as pd\nimport numpy as np\nimport pyemu\nimport flopy\n")
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
    pst.control_data.noptmax = 8
    pst.write(PST_NAME_KR)
    pyemu.helpers.run("pestchek {0}".format(PST_NAME_KR))
    pyemu.helpers.run("pestpp {0}".format(PST_NAME_KR.replace(".pst",".init.pst")))
    
    os.chdir("..")


def setup_pest_zn():
    setup_model(WORKING_DIR_ZN)
    os.chdir(WORKING_DIR_ZN)

    # setup hk parameters
    m = flopy.modflow.Modflow.load(MODEL_NAM,check=False)
    shutil.copy2(os.path.join('..',BASE_MODEL_DIR,"hk.zones"),"hk.zones")
    zone_arr = np.loadtxt("hk.zones",dtype=int)
    with open("hk_layer_1.ref.tpl",'w') as f:
        f.write("ptf ~\n")
        for i in range(m.nrow):
            for j in range(m.ncol):
                f.write(" ~  hk_z{0:02d}   ~".format(zone_arr[i,j]))
            f.write("\n")

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

    pst = _get_base_pst(m)

    
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["upgrade_augment"] = "false"
    pst.pestpp_options["lambdas"] = "0.1,1.0,10.0"
    
    pst.model_command = ["python forward_run.py"]
    pst.control_data.noptmax = 0

    pst.write(PST_NAME_ZN.replace(".pst",".init.pst"))

    with open("forward_run.py",'w') as f:
        f.write("import os\nimport shutil\nimport pandas as pd\nimport numpy as np\nimport pyemu\nimport flopy\n")
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
    pst.control_data.noptmax = 8
    pst.write(PST_NAME_ZN)
    pyemu.helpers.run("pestchek {0}".format(PST_NAME_ZN))
    pyemu.helpers.run("pestpp {0}".format(PST_NAME_ZN.replace(".pst",".init.pst")))
    
    os.chdir("..")

def setup_pest_gr():
    setup_model(WORKING_DIR_GR)
    os.chdir(WORKING_DIR_GR)

    # setup hk pilot point parameters
    m = flopy.modflow.Modflow.load(MODEL_NAM,check=False)
    with open("hk_layer_1.ref.tpl",'w') as f:
        f.write("ptf ~\n")
        for i in range(m.nrow):
            for j in range(m.ncol):
                f.write(" ~  hk_i{0:02d}_j{1:02d}   ~".format(i,j))
            f.write("\n")

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
    pst = _get_base_pst(m)

    # set some parameter attribs
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["upgrade_augment"] = "false"
    pst.pestpp_options["lambdas"] = "0.1,1.0,10.0"
    
    pst.model_command = ["python forward_run.py"]
    pst.control_data.pestmode = "regularization"
    pst.pestpp_options["n_iter_base"] = -1
    pst.pestpp_options["n_iter_super"] = 3
    pst.control_data.noptmax = 0

    # first order Tikhonov
    #cov = pyemu.helpers.pilotpoint_prior_builder(pst,{gs:[pp_file+".tpl"]},sigma_range=6)
    #pyemu.helpers.first_order_pearson_tikhonov(pst,cov)

    # zero order Tikhonov
    pyemu.helpers.zero_order_tikhonov(pst)

    pst.write(PST_NAME_GR.replace(".pst",".init.pst"))

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
    pst.control_data.noptmax = 8
    pst.write(PST_NAME_GR)
    pyemu.helpers.run("pestchek {0}".format(PST_NAME_GR))
    pyemu.helpers.run("pestpp {0}".format(PST_NAME_GR.replace(".pst",".init.pst")))
    
    os.chdir("..")

def setup_pest_pp():
    setup_model(WORKING_DIR_PP)
    os.chdir(WORKING_DIR_PP)
    # setup hk pilot point parameters
    m = flopy.modflow.Modflow.load(MODEL_NAM,check=False)
    pp_space = 4
    df_pp = pyemu.gw_utils.setup_pilotpoints_grid(ml=m,prefix_dict={0:["hk"]},
                                          every_n_cell=pp_space)
    pp_file = "hkpp.dat"

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

    pst = _get_base_pst(m)

    # set some parameter attribs
    par = pst.parameter_data
    par.loc[df_pp.parnme,"parval1"] = 5.0
    par.loc[df_pp.parnme,"parlbnd"] = 0.5
    par.loc[df_pp.parnme,"parubnd"] = 50.0
    par.loc[df_pp.parnme,"pargp"] = "hk"

    pst.model_command = ["python forward_run.py"]
    #pst.control_data.pestmode = "regularization"
    pst.pestpp_options["n_iter_base"] = -1
    pst.pestpp_options["n_iter_super"] = 3
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["upgrade_augment"] = "false"
    pst.pestpp_options["lambdas"] = "0.1,1.0,10.0"
    
    pst.control_data.noptmax = 0
    a = float(pp_space) * m.dis.delr.array[0] * 3.0
    v = pyemu.geostats.ExpVario(contribution=1.0,a=a)
    gs = pyemu.geostats.GeoStruct(variograms=[v],transform="log")
    ok = pyemu.geostats.OrdinaryKrige(gs,pyemu.gw_utils.pp_file_to_dataframe(pp_file))
    ok.calc_factors_grid(m.sr,var_filename="pp_var.ref")
    ok.to_grid_factors_file(pp_file+".fac")

    # first order Tikhonov
    #cov = pyemu.helpers.pilotpoint_prior_builder(pst,{gs:[pp_file+".tpl"]},sigma_range=6)
    #pyemu.helpers.first_order_pearson_tikhonov(pst,cov)

    # zero order Tikhonov
    #pyemu.helpers.zero_order_tikhonov(pst)

    pst.write(PST_NAME_PP.replace(".pst",".init.pst"))

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
    pst.control_data.noptmax = 8
    pst.write(PST_NAME_PP)
    pyemu.helpers.run("pestchek {0}".format(PST_NAME_PP))
    pyemu.helpers.run("pestpp {0}".format(PST_NAME_PP.replace(".pst",".init.pst")))
    
    os.chdir("..")


def build_prior_pp():
    v = pyemu.geostats.ExpVario(contribution=1.0,a=2000.0)
    gs = pyemu.geostats.GeoStruct(variograms=[v])
    pst_pp = pyemu.Pst(os.path.join(WORKING_DIR_PP,PST_NAME_PP))
    pp_tpl = os.path.join(WORKING_DIR_PP,[tpl for tpl in pst_pp.template_files if "pp" in tpl][0])
    cov_pp = pyemu.helpers.geostatistical_prior_builder(pst_pp,{gs:pp_tpl},sigma_range=6)
    return cov_pp

def build_prior_gr():
    v = pyemu.geostats.ExpVario(contribution=1.0,a=2000.0)
    gs = pyemu.geostats.GeoStruct(variograms=[v])
    pst_gr = pyemu.Pst(os.path.join(WORKING_DIR_GR,PST_NAME_GR))
    par = pst_gr.parameter_data
    hk_par = par.loc[par.parnme.apply(lambda x: x.startswith("hk")),:]
    hk_par.loc[:,"i"] = hk_par.parnme.apply(lambda x: int(x.split('_')[1].replace("i",'')))
    hk_par.loc[:,"j"] = hk_par.parnme.apply(lambda x: int(x.split('_')[2].replace("j",'')))
    m = flopy.modflow.Modflow.load(MODEL_NAM,model_ws=WORKING_DIR_GR,load_only=[])
    hk_par.loc[:,"x"] = hk_par.apply(lambda x: m.sr.xcentergrid[x.i,x.j],axis=1)
    hk_par.loc[:,"y"] = hk_par.apply(lambda x: m.sr.ycentergrid[x.i,x.j],axis=1)
    cov_gr = pyemu.helpers.geostatistical_prior_builder(pst_gr,struct_dict={gs:hk_par},sigma_range=6)
    return cov_gr


if __name__ == "__main__":
    setup_pest_un_bareass()
    #setup_pest_pp()
    #setup_pest_gr()
    #setup_pest_zn()
    #setup_pest_kr()
    #setup_pest_un()