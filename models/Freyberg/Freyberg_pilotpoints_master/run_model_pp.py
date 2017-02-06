import os
import platform
if 'window' in platform.platform().lower():
    pref = ''
else:
    pref = './'
import pyemu

pyemu.utils.gw_utils.fac2real("points1.dat","factors1.dat","hk.ref",lower_lim=1.0e-10)
os.system("{0}mf2005 freyberg.nam".format(pref))
os.system("{0}mp6 <mpath.in".format(pref))
os.system("python Process_output.py")
