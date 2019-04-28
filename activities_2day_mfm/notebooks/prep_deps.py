# todo: deal with linux bins!

import os
import platform
import shutil

py_dirs = [os.path.join("/","Users","jeremyw","Dev","pyemu","pyemu"),
           os.path.join("/","Users","jeremyw","Dev","flopy","flopy")]

pestpp_bin_dir = os.path.join("/","Users","jeremyw","Dev","pestpp","bin")

#mfnwt_bin_dir = os.path.join("/","Users","jeremyw","Dev","pestpp","benchmarks","test_bin")
mfnwt_bin_dir = os.path.join("..","..","bin")


if "linux" in platform.platform().lower():
    raise NotImplementedError()
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
    
if __name__ == "__main__":
    prep_for_deploy()  
    #prep_template(t_d="temp")  
