import os

os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace setup_transient_model.ipynb")
os.system("jupyter nbconvert --to pdf setup_transient_model.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace setup_transient_model.ipynb")

os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace setup_pest_interface.ipynb")
os.system("jupyter nbconvert --to pdf setup_pest_interface.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace setup_pest_interface.ipynb")

os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace process_obs_and_set_weights.ipynb")
os.system("jupyter nbconvert --to pdf process_obs_and_set_weights.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace process_obs_and_set_weights.ipynb")

os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace pestpp-ies_part1.ipynb")
os.system("jupyter nbconvert --to pdf pestpp-ies_part1.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace pestpp-ies_part1.ipynb")

os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace pestpp-ies_part2.ipynb")
os.system("jupyter nbconvert --to pdf pestpp-ies_part2.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace pestpp-ies_part2.ipynb")

os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace pestpp-ies_bmw.ipynb")
os.system("jupyter nbconvert --to pdf pestpp-ies_bmw.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace pestpp-ies_bmw.ipynb")


os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace prior_montecarlo.ipynb")
os.system("jupyter nbconvert --to pdf prior_montecarlo.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace prior_montecarlo.ipynb")

os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace pestpp-glm_part1.ipynb")
os.system("jupyter nbconvert --to pdf pestpp-glm_part1.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace pest_glm_part1.ipynb")

os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace dataworth.ipynb")
os.system("jupyter nbconvert --to pdf dataworth.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace dataworth.ipynb")

os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace pestpp-glm_part2.ipynb")
os.system("jupyter nbconvert --to pdf pestpp-glm_part2.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace pest_glm_part2.ipynb")


os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace pestpp-opt.ipynb")
os.system("jupyter nbconvert --to pdf pestpp-opt.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace pestpp_opt.ipynb")

os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace pestpp-ies_the_wrong_way.ipynb")
os.system("jupyter nbconvert --to pdf pestpp-ies_the_wrong_way.ipynb")
os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace pestpp-ies_the_wrong_way.ipynb")

