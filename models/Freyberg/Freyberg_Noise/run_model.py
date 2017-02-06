import os
import platform
if 'window' in platform.platform().lower():
    pref = ''
else:
    pref = './'

os.system('{0}mf2005 freyberg.noise.nam'.format(pref))
os.system('{0}python Process_output.py'.format(pref))
