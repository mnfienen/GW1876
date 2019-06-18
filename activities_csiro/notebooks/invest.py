import os
import matplotlib.pyplot as plt
import numpy as np


def check_arr():
    d1 = os.path.join("template","arr_mlt")
    d2 = os.path.join("template1","arr_mlt")

    ref_files1 = [f for f in os.listdir(d1)]# if f.endswith(".ref")]
    ref_files2 = [f for f in os.listdir(d2)]# if f.endswith(".ref")]

    for ref_file1 in ref_files1:
        if ref_file1 not in ref_files2:
            print("missing",ref_file1)
            continue
        arr1 = np.loadtxt(os.path.join(d1,ref_file1))
        arr2 = np.loadtxt(os.path.join(d2,ref_file1))
        diff = arr1 - arr2
        if np.abs(diff).max() > 1e-6:
            print(ref_file1,np.abs(diff).max())
            diff = np.ma.masked_where(np.abs(diff)<1.0e-6,diff)
            ax = plt.subplot(111,aspect="equal")
            a = ax.imshow(diff)
            ax.set_title(ref_file1)
            plt.colorbar(a)
            plt.show()


def check_factors():
    d1 = os.path.join("template","arr_mlt")
    d2 = os.path.join("template1","arr_mlt")

    ref_files1 = [f for f in os.listdir(d1) if "pp.dat" in f]
    ref_files2 = [f for f in os.listdir(d2) if "pp.dat" in f]

    for ref_file1 in ref_files1:
        if ref_file1 not in ref_files2:
            print("missing",ref_file1)
            continue
        df1 = pd.read_csv(os.path.join(d1,ref_file1),delim_whitespace=True)
        df2 = pd.read_csv(os.path.join(d2,ref_file1),delim_whitespace=True)
        
        diff = df1.iloc[:,-1] - df2.iloc[:,-1]
        if diff.max() > 1e-6:
            print(ref_file1,diff.max())
        
def debug_f2r():
    import pyemu
    pp_file = os.path.join("template","hk1pp.dat")
    fac_file = os.path.join("template","pp_k1_general_zn.fac")
    arr = pyemu.geostats.fac2real(pp_file,factors_file=fac_file,out_file=None)
    print(arr)

if __name__ == "__main__":
    #check_arr()
    #check_factors()
    debug_f2r()