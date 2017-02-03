import shutil
import os

model_cases = ['10par_xsec','Freyberg']

exe_files = [f for f  in os.listdir(os.getcwd()) if not f.endswith('.py')]

for ccase in model_cases:
    currd = os.path.join('..','models',ccase)
    for d in os.listdir(currd):
        currdd = os.path.join(currd,d)
        if os.path.isdir(currdd):
            existing_exes = [os.path.join(currdd,f) for f in os.listdir(currdd) if f in exe_files]
            for eexe in existing_exes:
                os.remove(eexe)
            print ('Copying exes to {0}'.format(os.path.abspath(os.path.join(currd,d))))
            [shutil.copy2(cf,os.path.join(currd,d,cf)) for cf in exe_files]
