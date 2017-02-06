import pyemu
pst = pyemu.Pst("freyberg_pp.pst")
pst.parrep("freyberg_pp.bpa")
pst.control_data.noptmax = 0
pst.write("freyberg_pp_final.pst")