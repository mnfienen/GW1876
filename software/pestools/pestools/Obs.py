__author__ = 'aleaf'

import sys
sys.path.append('../pst_tools')
from pst import *
from res import *
import pandas as pd

import numpy as np

class Obs(Pest):
    """
    Class for working with observations, especially with observation weighting.

    Goal is to facilitate:
    * replacing observation data in pest control file with a new observation dataset from an external file
    * interactively adjust weighting and visualize changes to the objective function


    Attributes
    ----------

    """

    def __init__(self, basename):

        Pest.__init__(self, basename)

        self._read_obs_data()

        self._new_obs_data = pd.DataFrame()

        # get residuals information from PEST run
        self.res = Res(basename + '.res')

        # copy observation data for weight adjustment
        self.df = self.res.df.copy()

        # get list of groups that aren't regularisation
        self.groups = np.unique(self.obsdata.OBGNME)


    def plot_objective_contrib(self):
        self.res.plot_objective_contrib(self.df)

        return


    def objective_contrib(self):
        self.res.objective_contrib(self.df)

        return

    def mikes_weighting_routine(self):
        # Not implemented yet
        return


    def widget_method_for_group_weighting(self):
        # ?? Not implemented yet
        return