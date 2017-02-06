# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 13:35:24 2015

@author: egc
"""
import os
import numpy as np
import pandas as pd
from .mat_handler import jco as Jco
from .mat_handler import cov as Cov
from .pst_handler import pst as Pst
from .Cor import Cor



class Pest(object):
    """
    base class for PEST run
    contains run name, folder and some control data from PEST control file
    also has methods to read in parameter and observation data from PEST control file

    could also be a container for other global settings
    (a longer run name to use in plot titles, etc)

    basename : string
    pest basename or pest control file (includes path)
    """

    def __init__(self, basename, obs_info_file=None, par_info_file=None,
                 name_col='Name', x_col='X', y_col='Y', type_col='Type',
                 error_col='Error', basename_col='basename', datetime_col='datetime', group_cols=[],
                 obs_info_kwds={}):

        self.basename = os.path.split(basename)[-1].split('.')[0]
        self.run_folder = os.path.split(basename)[0]
        if len(self.run_folder) == 0:
            self.run_folder = os.getcwd()

        self.pstfile = os.path.join(self.run_folder, self.basename + '.pst')
        
        # Thinking this will get pass along later to the Res class or similar
        self.obs_info_file = obs_info_file
        if obs_info_file is not None:
            self._read_obs_info_file(obs_info_file, name_col=name_col,
                                     x_col=x_col, y_col=y_col, type_col=type_col,
                                     error_col=error_col, basename_col=basename_col,
                                     datetime_col=datetime_col, group_cols=group_cols,
                                     obs_info_kwds=obs_info_kwds)
        else:
            self.obsinfo = pd.DataFrame()

        self.par_info_file = par_info_file
        if par_info_file is not None:
            self._read_par_info_file(par_info_file)
        else:
            self.parinfo = pd.DataFrame()

    def IdentPar(self, jco=None, par_info_file=None):
        '''
        IdentPar class
        '''
        from identpar import IdentPar
        if jco is None:
            jco = self.pstfile.strip('pst')+'jco'
        identpar = IdentPar(jco, par_info_file)
        return identpar
    
    @property    
    def _jco(self):
        '''
        Matrix class of jco
        '''
        jco = Jco()
        jco.from_binary(os.path.splitext(self.pstfile)[0]+'.jco')
        return jco
    @property
    def jco_df(self):
        '''
        DataFrame of jco
        '''
        jco_df = self._jco.to_dataframe()
        return jco_df

    @property
    def pst(self):
        '''
        Pst Class
        '''
        pst = Pst(self.pstfile)
        return pst
        

    def ParSen(self, **kwargs):
        '''
        ParSen class
        '''
        from parsen import ParSen
        parsen = ParSen(basename=self.pstfile, jco_df = self.jco_df,
                        res_df = self.res_df, 
                        parameter_data = self.parameter_data, **kwargs)
        return parsen
        
    def ObSen(self, **kwargs):
        '''
        ObSen class
        '''
        from obsen import ObSen
        obsen = ObSen(basename=self.pstfile, jco_df = self.jco_df,
                        res_df = self.res_df, 
                        parameter_data = self.parameter_data, **kwargs)
        return obsen
    @property    
    def rmr(self):
        '''
        rmr class
        '''
        from rmr import Rmr
        rmr = Rmr(basename = self.pstfile)
        
        return rmr
        
    
    def res(self, res_file, obs_info_file = None):
        '''
        Res Class
        
        Parameters
        ----------
        res_extension : str
           The extension of the residual file to load.  Assumes basename
           from Pest.basename.  For exmaple 'rei' or 'res'
        '''
        from res import Res
        #res_file = self.pstfile.rstrip('pst')+res_extension
        res = Res(res_file, obs_info_file)

        return res
    
    @property
    def res_df(self):
        '''
        Residual DataFrame
        '''
        res = self.pst.res
        return res

    @property
    def par(self, **kwargs):
        from par import Par
        '''
        DataFrame of data from .par file
        '''
        par = Par(basename = self.pstfile)
        return par

    @property
    def parameter_data(self):
        '''
        DataFrame of parameter data in .pst file
        '''
        parameter_data = self.pst.parameter_data
        return parameter_data
        
    @property
    def observation_data(self):
        '''
        DataFrame of observation data
        '''
        observation_data = self.pst.observation_data
        observation_data.index = observation_data.obsnme
        return observation_data

    @property
    def obs_groups(self):
        '''
        List of observation groups
        '''
        obs_groups = self.pst.obs_groups
        return obs_groups
        
    @property
    def _cov(self):
        weights = self.res_df['weight'].values
        phi = self.pst.phi
        pars = self._jco.col_names
        
        # Calc Covariance Matrix
        # See eq. 2.17 in PEST Manual
        # Note: Number of observations are number of non-zero weighted observations
        q = np.diag(np.diag(np.tile(weights**2, (len(weights), 1))))
        cov = np.dot((phi/(np.count_nonzero(weights)-len(pars))),
                     (np.linalg.inv(np.dot(np.dot(self._jco.x.T, q),self._jco.x))))
        cov = Cov(x=cov, names = pars)
        return cov

    @property
    def cov_df(self):
        cov_df = self.cov.to_dataframe()
        return cov_df
        
    @property
    def cor(self):
        return Cor(self._cov)

    def _read_obs_info_file(self, obs_info_file, name_col='Name', x_col='X', y_col='Y', type_col='Type',
                            error_col='Error', basename_col='basename', datetime_col='datetime', group_cols=[],
                            obs_info_kwds={}):
            """Bring in ancillary observation information from csv file such as location and measurement type
            """
            self.obsinfo = pd.read_csv(obs_info_file, index_col=name_col, **obs_info_kwds)
            self.obsinfo.index = [n.strip().lower() for n in self.obsinfo.index]
    
            # remap observation info columns to default names
            self.obsinfo.rename(columns={x_col: 'X', y_col: 'Y', type_col: 'Type', error_col: 'Error'}, inplace=True)
    
            # make a dataframe of observation type for each group
            if 'Type' in self.obsinfo.columns:
                #self._read_obs_data()
                # join observation info to 'obsdata' so that the type and group for each observation are listed
                self.obsinfo = self.obsinfo.join(self.observation_data['obgnme'], lsuffix='', rsuffix='1', how='inner')
                self.obsinfo.rename(columns={'obgnme': 'Group'}, inplace=True)
                '''
                self._obstypes = self.obsinfo.drop_duplicates(subset='Group').ix[:, ['Group', 'Type']]
                self._obstypes.index = self._obstypes.Group
                self._obstypes = self._obstypes.drop('Group', axis=1)
                '''
    def _read_par_info_file(self, par_info_file, name_col='Name', x_col='X', y_col='Y', type_col='Type',
                            basename_col='basename', datetime_col='datetime', group_cols=[], **kwds):
            """Bring in ancillary parameter information from csv file such as location and parameter type
            """
            self.parinfo = pd.read_csv(par_info_file, index_col=name_col, **kwds)
            self.parinfo.index = [n.lower() for n in self.parinfo.index]

            # remap observation info columns to default names
            self.parinfo.rename(columns={x_col: 'X', y_col: 'Y', type_col: 'Type', 'foo': 'foo'}, inplace=True)
          
if __name__ == '__main__':
    p = Pest(r'C:\Users\egc\Desktop\identpar_testing\ppestex\test')
#    jco2 = Matrix()
#    jco2.from_binary(r'C:\Users\egc\Desktop\identpar_testing\ppestex\test.jco')
#    jco3 = Jco()
#    jco3.from_binary(r'C:\Users\egc\Desktop\identpar_testing\ppestex\test.jco')
    parsen = Pest(r'C:\Users\egc\pest_tools-1\cc\columbia')