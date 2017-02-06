import numpy as np
import pyemu

# only K
obs_data = np.loadtxt("10par_xsec.hds.init").reshape(20,1)
obs_wghts = np.zeros_like(obs_data)
obs_wghts[[3,5]] = 1.0
pst = pyemu.pst_utils.pst_from_io_files(["hk_Layer_1.ref.tpl"],["hk_Layer_1.ref"],["10par_xsec.hds.ins"],["10par_xsec.hds"])
pst.observation_data.obsval = obs_data
pst.observation_data.weight = obs_wghts
pst.prior_information = pst.null_prior
pst.parameter_data.loc[:,"parval1"] = 2.5
pst.control_data.pestmode = "estimation"
pst.write("k.pst")


#K and well
obs_data = np.loadtxt("10par_xsec.hds.init").reshape(20,1)
obs_wghts = np.zeros_like(obs_data)
obs_wghts[[3,5]] = 1.0
pst = pyemu.pst_utils.pst_from_io_files(["hk_Layer_1.ref.tpl","10par_xsec.wel.tpl"],
	["hk_Layer_1.ref","10parxsec.wel"],["10par_xsec.hds.ins"],["10par_xsec.hds"])
pst.parameter_data.loc[:,"parval1"] = [2.5,0.5,1.0]
pst.observation_data.obsval = obs_data
pst.observation_data.weight = obs_wghts
pst.prior_information = pst.null_prior
pst.control_data.pestmode = "estimation"
pst.write("k_wel.pst")