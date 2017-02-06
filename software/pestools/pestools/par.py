# -*- coding: utf-8 -*-
"""
Created on Wed Sep 09 13:16:06 2015

@author: egc
"""

import numpy as np
import pandas as pd
import os
from pest import Pest


class Par(object):

    #def __init__(self, basename=None, par_set=None):
    def __init__(self, basename=None, par_set=None,  obs_info_file=None, name_col='Name',
                 x_col='X', y_col='Y', type_col='Type',
                 basename_col='basename', datetime_col='datetime', group_cols=[],
                 obs_info_kwds={}, 
                **kwds):

        ''' Create Par class that works with data from a .par file

        Parameters
        ----------
        basename : str, optional
            basename for PEST control file, if full path not provided the 
            current working directory is assumed.
            
        par_set : Int, optional
            .par file number if multiple .par files writen with PEST.  If not
            provided then uses basename.par for the par file.  If provided uses
            basename.par.par_set as the par file
            
        Attributes
        ----------
        df : Pandas DataFrame
            DataFrame of par file.  Index entries of the DataFrame
            are the parameter names.  The DataFrame has two columns:
            1) Parameter Group and 2) Sensitivity
            
        '''

        
        if basename is not None:
            self.basename = os.path.split(basename)[-1].split('.')[0]
            self.directory = os.path.split(basename)[0]
            if len(self.directory) == 0:
                self.directory = os.getcwd() 
                
        if par_set is not None:
            self.par_file = os.path.join(self.directory, self.basename + '.par.%d' % (par_set))
        else:
            self.par_file = os.path.join(self.directory, self.basename + '.par')
         
        self.df = self.load_par_file()
        
        # Expose the Pest class for convience but not all attributes make sense
        # when dealing with the Par class alone so make private        
        self._Pest = Pest(basename, obs_info_file=obs_info_file, name_col=name_col,
                                           x_col=x_col, y_col=y_col, type_col=type_col,
                                           basename_col=basename_col, datetime_col=datetime_col,
                                           group_cols=group_cols, obs_info_kwds=obs_info_kwds)
        
    def load_par_file(self):        
        f = open(self.par_file, 'r')
        header = f.readline()
        df = pd.read_csv(f, header = None, names = ['parnme', 'parval', 'scale', 'offset'],
                         sep="\s+")
        df.index = df.parnme
        return df
    def parval(self, parnme):
        parval = self.df.ix[parnme]['parval']
        return parval
    @property    
    def at_bounds(self):
        bound_df = pd.merge(self.df, self._Pest.parameter_data, on = 'parnme')[['parnme','parval','parlbnd', 'parubnd']]
        bound_df['at_upper'] = (bound_df['parubnd']-bound_df['parval']) == 0.0
        bound_df['at_lower'] = (bound_df['parval']-bound_df['parlbnd']) == 0.0
        bound_df = bound_df[(bound_df['at_lower'] == True) | (bound_df['at_upper'] == True)].sort('at_upper', ascending = False)
        return bound_df
        
            
            