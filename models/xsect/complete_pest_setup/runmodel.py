import platform
import os

try:
    os.remove('10par_xsec.hds')
except:
    pass
if 'window' in platform.platform().lower():
	mf = 'mf2005'
else:
	mf = './mf2005'
os.system('{0} 10par_xsec.nam'.format(mf))
