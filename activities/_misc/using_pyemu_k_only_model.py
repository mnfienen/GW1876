
# coding: utf-8

# In[1]:

import os
import flopy
import pyemu


# In[2]:

# the path to where the jacobian matrix is in the file system
jco_path = os.path.join("..","FreybergModelForClass","Freyberg_K_only","freyberg.jcb")
# and path to the pest control file
pst_path = jco_path.replace(".jcb",".pst")


# In[4]:

# load the control file
pst = pyemu.Pst(pst_path)
# get the forecast names
forecast_names = pst.pestpp_options["forecasts"].split(',')
sc = pyemu.Schur(jco=jco_path)#,forecasts=forecast_names)


# In[5]:

sc.get_parameter_summary()


# In[6]:

sc.get_forecast_summary()


# In[ ]:



