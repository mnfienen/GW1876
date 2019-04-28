jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace setup_pest_interface.ipynb
jupyter nbconvert --to pdf setup_pest_interface.ipynb
jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace prior_montecarlo.ipynb
jupyter nbconvert --to pdf prior_montecarlo.ipynb
jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace pestpp-glm.ipynb
jupyter nbconvert --to pdf pestpp-glm.ipynb
jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace pestpp-ies.ipynb
jupyter nbconvert --to pdf pestpp-ies.ipynb
jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace pestpp-opt.ipynb
jupyter nbconvert --to pdf pestpp-opt.ipynb
