import platform
import os

if 'window' in platform.platform().lower():
	mf = 'mf2005'
else:
	mf = './mf2005'
os.system('{0} 10par_xsec.nam'.format(mf))
