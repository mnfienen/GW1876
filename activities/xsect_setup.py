import os
import sys
import shutil
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import pyemu

EXE_DIR = os.path.join("..","..","bin")
WORKING_DIR = "xsect_model"
BASE_MODEL_DIR = os.path.join("..","..","models","xsect","complete_pest_setup")
PST_NAME = 'k_wel_reg.pst'
NUM_SLAVES = 10
NUM_STEPS_RESPSURF = 25


def setup_model():

    if "window" in platform.platform().lower():
        exe_files = [f for f in os.listdir(EXE_DIR) if f.endswith('exe')]
    else:
        exe_files = [f for f in os.listdir(EXE_DIR) if not f.endswith('exe')]

    mod_files = [f for f in os.listdir(BASE_MODEL_DIR)]
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)
    for exe_file in exe_files:
        shutil.copy2(os.path.join(EXE_DIR,exe_file),os.path.join(WORKING_DIR,exe_file))
    for mod_file in mod_files:
        shutil.copy2(os.path.join(BASE_MODEL_DIR, mod_file), os.path.join(WORKING_DIR, mod_file))

