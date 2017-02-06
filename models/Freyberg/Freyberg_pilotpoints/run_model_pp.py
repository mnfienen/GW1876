import os
import platform
import pyemu
if 'window' in platform.platform().lower():
    pref = ''
else:
    pref = './'
for f in ["freyberg.hyd.bin","freyberg.heads","freyberg.list","freyberg.mpenpt"]:
	try:
		os.remove(f)
	except:
		pass

pyemu.utils.gw_utils.fac2real("points1.dat","factors1.dat","hk.ref",lower_lim=1.0e-10)
os.system("{0}mf2005 freyberg_pp.nam".format(pref))
os.system("{0}mp6 <mpath.in".format(pref))
os.system("python Process_output.py")
