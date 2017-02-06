
# coding: utf-8

# In[133]:

from __future__ import print_function
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import flopy
import pyemu


# In[134]:

#model_ws = os.path.join("..","model")
#nam_file = "10par_xsec.nam"
model_ws = os.path.join("..","FreybergModelForClass","Freyberg_Truth")
nam_file = "freyberg.truth.nam"
ml = flopy.modflow.Modflow.load(nam_file,model_ws=model_ws,check=False,verbose=True,
                                forgive=False)


# In[135]:

variance_fraction = {"flux":0.1,"stage":0.05,"cond":0.5}


# In[136]:

# wells
wdata = ml.wel.stress_period_data.array["flux"]
noise = np.random.standard_normal(wdata.shape)
# scale by the variance of each pumping well
noise *= wdata * variance_fraction["flux"]
wdata += noise
wdata


# In[137]:

wdata = flopy.utils.MfList.masked4D_arrays_to_stress_period_data(flopy.modflow.ModflowWel.
                                                                 get_default_dtype(),{"flux":wdata})
wel = flopy.modflow.ModflowWel(ml,stress_period_data=wdata)


# In[142]:

array_dict = ml.riv.stress_period_data.array
for prop in ["cond","stage"]:    
    noise = np.random.standard_normal(array_dict[prop].shape)
    #scale by the variance of each pumping well
    noise *= array_dict[prop] * variance_fraction[prop]
    array_dict[prop] += noise
array_dict = flopy.utils.MfList.masked4D_arrays_to_stress_period_data(flopy.modflow.ModflowRiv.
                                                                 get_default_dtype(),array_dict)
riv = flopy.modflow.ModflowRiv(ml,stress_period_data=array_dict)


# In[143]:

print(ml.rech)


# In[144]:

ml.name = "freyberg.noise"
ml.write_input()


# In[ ]:



