import os

nb_files = [f for f in os.listdir(".") if f.lower().endswith(".ipynb")]
for nb_file in nb_files:
    os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {0}".format(nb_file))

for d in [".","intro_to_fosm","bayes_background","singular_value_decomposition"]:
	nb_files = [f for f in os.listdir(d) if f.lower().endswith(".ipynb")]
	for nb_file in nb_files:
	    os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {0}".format(os.path.join(d,nb_file)))
