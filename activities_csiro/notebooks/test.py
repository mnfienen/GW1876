import os
import pyemu

pst = pyemu.Pst(os.path.join("master_glm","freyberg_pp.pst"))
cov = pyemu.Cov.from_binary(os.path.join("master_glm","prior.jcb"))

pyemu.helpers.first_order_pearson_tikhonov(pst,cov)