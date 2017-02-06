__author__ = 'aleaf'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyemu import ErrVar
from .pest import Pest
from . import plots

class IdentPar:

    def __init__(self, jco, par_info_file=None):
        """Computes parameter identifiability for a PEST jco file,
        using the ErrVar class in pyemu (https://github.com/jtwhite79/pyemu)
        """

        self._Pest = Pest(jco, par_info_file=par_info_file)
        self.parinfo = self._Pest.parinfo

        self.la = ErrVar(jco)
        self.parinfo = None
        if par_info_file is not None:
            self.parinfo = pd.read_csv(par_info_file, index_col='Name')
        self.ident_df = None

    def plot_singular_spectrum(self):
        """see http://nbviewer.ipython.org/github/jtwhite79/pyemu/blob/master/examples/error_variance_example.ipynb
        """
        s = self.la.qhalfx.s

        figure = plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
        ax.plot(s.x)
        ax.set_title("singular spectrum")
        ax.set_ylabel("power")
        ax.set_xlabel("singular value")
        ax.set_xlim(0,300)
        ax.set_ylim(0,10)
        plt.show()

    def get_identifiability_dataframe(self, nsingular):

        self.ident_df = self.la.get_identifiability_dataframe(nsingular)
        if self.parinfo is not None:
            self.ident_points = pd.DataFrame({'ident_sum': self.ident_df.sum(axis=1)}).join(self.parinfo)
            self.ident_points.ident_sum = [i/2.0 for i in self.ident_points.ident_sum]


    def plot_bar(self, nsingular=None, nbars=20):
        """Computes a stacked bar chart showing the most identifiable parameters
        at a given number of singular values

        Parameters:
        -----------
        nsingular:
            number of singular values to include

        nbars:
            number of parameters (bars) to include in bar chart
        """
        if nsingular is not None:
            self.get_identifiability_dataframe(nsingular)

        plot_obj = plots.IdentBar(self.ident_df, nsingular=nsingular, nbars=nbars)
        plot_obj.generate()
        plot_obj.draw()
        
        return plot_obj.fig, plot_obj.ax

    def plot_spatial(self, nsingular=None,
                     groupinfo={},
                     group_col=None,
                     legend_kwds={}, **kwds):

        if nsingular is not None:
            self.get_identifiability_dataframe(nsingular)


        """
        Make a spatial plot of total parameter identifiabilities

        Parameters
        ----------


        Notes
        ------

        """


        # if no group column is argued, default to all spatial parameters in one group
        df = self.ident_points
        if group_col is None:
            group_col = 'partype'
            df.loc[np.isnan(df.X), group_col] = 'lumped'
            df.loc[~np.isnan(df.X), group_col] = 'spatial'


        plot_obj = plots.SpatialPlot(df, 'X', 'Y', 'ident_sum', groupinfo, group_col=group_col,
                                     colorby='k', legend_values=[0, 0.5, 1],
                                     legend_kwds=legend_kwds, **kwds)
        plot_obj.generate()
        plot_obj.draw()

        return plot_obj