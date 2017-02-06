# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 22:54:50 2015

@author: egc
"""
import numpy as np
import pandas as pd
import os
from . import plots
from .mat_handler import matrix as Matrix
from .pst_handler import pst as Pst


class Cor(object):
    def __init__(self, cov):
        '''
        Parameters
        ----------
        cov : Cov Matrix class
  
        Attributes
        ----------
        df : DataFrame
            DataFrame of the correlation coefficient matrix

        Methods
        --------
        plot_heatmap
        pars
        '''

        #cov = self._cov().values
        d = np.diag(cov.x)
        cor = cov.x/np.sqrt(np.multiply.outer(d, d))
        
        self.matrix = Matrix(x=cor, row_names = cov.col_names, col_names = cov.col_names)
        # Put into dataframe
        self.df = self.matrix.to_dataframe()


         
        

    def pars(self, par_list, inplace = False):
        ''' Reduce the correlation coefficient matrix to select parameters
        Parameters
        ----------
        par_list : list
            list of parameters to show correlation coefficient matrix for
            
        inplace : False, True
            If False return a smaller DataFrame of Cor.df
            If True, change DataFrame of Cor.df inplace
           

        Returns
        -------
        df : DataFrame
            DataFrame of the correlation coefficient matrix with only select
            parameters
        '''
        reduced_matrix = self.df.loc[par_list][par_list]
        if inplace == False:
            return reduced_matrix
        if inplace == True:
            self.df = reduced_matrix

    def plot_heatmap(self, label_rows=True, label_cols=True, par_list=None, **kwds):
        ''' Plot correlation coefficient matrix

        Parameters
        ----------
        label_rows : bol, optional
            label the rows. Default is True.  For large matrices it is often
            cleaner to set False

        label_cols : bol, optional
            label the columns. Default is True.  For large matrices it is often
            cleaner to set False

        par_list : list, optional
            list of parameters to show correlation coefficient matrix for.
            Useful for large matrices

        Returns
        -------
        Matplotlib plot
            Heatmap (pcolormesh) of correlation coefficient matrix
        '''
        if par_list is None:
            df = self.df
        else:
            df = self.pars(par_list)
        plot_obj = plots.HeatMap(df, label_rows=label_rows,
                                 label_cols=label_cols, vmin=-1.0,
                                 vmax=1.0, **kwds)
        plot_obj.generate()
        plot_obj.draw()
        return plot_obj.fig, plot_obj.ax