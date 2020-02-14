# todo: deal with linux bins!

import os
import platform
import shutil

py_dirs = [os.path.join("/","Users","jwhite","Dev","pyemu","pyemu"),
           os.path.join("/","Users","jwhite","Dev","flopy","flopy")]

pestpp_bin_dir = os.path.join("/","Users","jwhite","Dev","pestpp","bin")

#mfnwt_bin_dir = os.path.join("/","Users","jeremyw","Dev","pestpp","benchmarks","test_bin")
mfnwt_bin_dir = os.path.join("..","..","bin")


if "linux" in platform.platform().lower():
    #raise NotImplementedError()
    os_d = "linux"
elif "darwin" in platform.platform().lower():
    os_d = "mac"
else:
    os_d = "win"


def prep_for_deploy():
    for py_dir in py_dirs:
        dest_dir = os.path.split(py_dir)[-1]
        assert os.path.exists(py_dir),py_dir
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(py_dir,dest_dir)
    
    dest_dir = os.path.split(pestpp_bin_dir)[-1]
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(pestpp_bin_dir,dest_dir)

    # deal with the iwin crap:
    iwin_dir = os.path.join("bin","iwin")
    win_dir = os.path.join("bin","win")
    
    shutil.rmtree(win_dir)
    os.makedirs(win_dir)
    files = os.listdir(iwin_dir)
    for f in files:
        shutil.copy2(os.path.join(iwin_dir,f),os.path.join(win_dir,f[1:]))
    shutil.rmtree(iwin_dir)

    # forward model bins

    for f,d in zip(["mfnwt.exe","mfnwt"],["win","mac"]):
        shutil.copy2(os.path.join(mfnwt_bin_dir,f),os.path.join("bin",d,f))
    for f,d in zip(["mp6.exe","mp6"],["win","mac"]):
        shutil.copy2(os.path.join(mfnwt_bin_dir,f),os.path.join("bin",d,f))



def prep_template(t_d="template"):
    for d in ["pyemu","flopy"]:
        if os.path.exists(os.path.join(t_d,d)):
            shutil.rmtree(os.path.join(t_d,d))
        shutil.copytree(d,os.path.join(t_d,d))
    files = os.listdir(os.path.join("bin",os_d))
    for f in files:
        if os.path.exists(os.path.join(t_d,f)):
            os.remove(os.path.join(t_d,f))
        shutil.copy2(os.path.join("bin",os_d,f),os.path.join(t_d,f))
    

def prep_forecasts(t_d="template_history",pst_name="freyberg.pst",
                   b_d=os.path.join("..","base_model_files")):
    import pyemu
    import flopy
    import pandas as pd

    m = flopy.modflow.Modflow.load("freyberg.nam",model_ws=t_d,check=False,load_only=["DIS"])
    pst = pyemu.Pst(os.path.join(t_d,pst_name))
    fore_csv = os.path.join(b_d,"forecast_truth.csv")
    assert os.path.exists(fore_csv)
    fore_df = pd.read_csv(fore_csv,index_col=0)
    obs = pst.observation_data
    obs_fore = obs.loc[pst.forecast_names,:].copy()
    print(obs_fore)
    for tag in ["hds","fa_hw","fa_tw","part_time","part_status"]:
        truth_val = fore_df.loc[fore_df.index.map(lambda x: tag in x),"obsval"]
        assert truth_val.shape[0] == 1,truth_val
        truth_val = truth_val.values[0]
        obs_name = obs_fore.loc[obs_fore.index.map(lambda x: tag in x),"obsnme"]
        assert obs_name.shape[0] == 1,obs_name
        obs.loc[obs_name,"obsval"] = truth_val
    print(obs.loc[pst.forecast_names,:])
    pst.write(os.path.join(t_d,pst_name))

if __name__ == "__main__":
    prep_forecasts()
    #prep_for_deploy()  
    #prep_template(t_d="temp")  
