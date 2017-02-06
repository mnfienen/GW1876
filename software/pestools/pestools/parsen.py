# -*- coding: utf-8 -*-
"""


@author: egc
"""
import numpy as np
import pandas as pd
import os
from . import plots
from .mat_handler import jco as Jco
from .pst_handler import pst as Pst



class ParSen(object):

    def __init__(self, basename=None, parameter_data=None, res_df=None, 
                 jco_df=None, drop_regul=False, drop_groups=None, 
                 keep_groups=None, keep_obs=None, remove_obs=None):

        ''' Create ParSen class

        Parameters
        ----------
        basename : str, optional
            basename for PEST control file, if full path not provided the 
            current working directory is assumed.  Optional but must be provided
            if any of parameter_data, res_df, or jco_df are not provided.
            
        parameter_data : DataFrame, optional
            Pandas DataFrame of the paramter data from a .pst file.  If not
            provided it will be read in based on the base name of pest file
            provided.
            
        jco_df : DataFrame, optional
            Pandas DataFrame of the jacobian. If not provided then it will be
            read in based on base name of pest file provided. Providing a
            jco_df offers some efficiencies if working interactively.
            Otherwise the jco is read in every time ParSen class is initialized.
            
            
        res_df : DataFrame, optional
            Residual DataFrame used to define the weights to 
            calculate the parameter sensitivity.  Providing a
            res_df offers some efficiencies if working interactively.
            If not provided it will look for basename+'.res'.  
            Weights are not taken from PEST control file
            (.pst) because regularization weights in PEST conrtrol file do
            not reflect the current weights.

        drop_regul: {False, True}, optional
            Flag to drop regularization information in calculating parameter
            sensitivity.  Will set weight to zero for all observations with
            'regul' in the observation group name

        drop_groups: list, optional
            List of observation groups to drop when calculating parameter
            sensitivity.  If all groups are part of regularization it may
            be easier to use the drop_regul flag

        keep_groups: list, optional
            List of observation groups to include in calculating parameter
            sensitivity.  Sometimes easier to use when looking at sensitivity
            to a single, or small number, or observation groups

        keep_obs: list, optional
            List of observations to include in calculating parameter
            sensitivity.  If keep_obs != None then weights for all observations
            not in keep_obs will be set to zero.

        remove_obs: list, optional
            List of observations to remove in calculating parameter
            sensitivity.  If remove_obs != None then weights for all
            observations in remove_obs will be set to zero.

        Attributes
        ----------
        df : Pandas DataFrame
            DataFrame of parameter sensitivity.  Index entries of the DataFrame
            are the parameter names.  The DataFrame has two columns:
            1) Parameter Group and 2) Sensitivity

        Methods
        -------
        plot()
        tail()
        head()
        par()
        group()
        sum_group()
        plot_sum_group()
        plot_mean_group()



        Notes
        ------

        '''
        if basename is not None:
            self.basename = os.path.split(basename)[-1].split('.')[0]
            self.directory = os.path.split(basename)[0]
            if len(self.directory) == 0:
                self.directory = os.getcwd()   


        if jco_df is None:
            jco_file = os.path.join(self.directory, self.basename + '.jco')
            jco = Jco()
            jco.from_binary(jco_file)
            self.jco_df = jco.to_dataframe()
        else:
            self.jco_df = jco_df
        
        if res_df is None:
            res_file = os.path.join(self.directory, self.basename + '.res')
            pst = Pst(filename=None, load=False, resfile=res_file)
            self.res_df = pst.load_resfile(res_file)
        else:
            self.res_df = res_df
        # Set index of res_df

        if 'Name' in self.res_df.columns:
            self.res_df.columns = [i.lower() for i in self.res_df.columns]
        self.res_df.set_index('name', drop=False, inplace = True)

        
        if parameter_data is None:
            pst_file = os.path.join(self.directory, self.basename + '.pst')
            pst = Pst(pst_file, load=True, resfile = None)
            self.parameter_data = pst.parameter_data
        else:
            self.parameter_data = parameter_data

       
        # Build pars_dict
        # key is PARNME value is PARGP
        self._pars_dict = {}
        for index, row in self.parameter_data.iterrows():
            self._pars_dict[row['parnme'].lower()] = row['pargp'].lower()

        # Build _obs_data
        weights = []
        ob_groups = []
        obs = []
        for ob in self.jco_df.index:
            weight = self.res_df.loc[ob.lower()]['weight']
            ob_group = self.res_df.loc[ob.lower()]['group']
            weights.append(weight)
            ob_groups.append(ob_group)
            obs.append(ob)
        self._obs_data = pd.DataFrame({'OBSNME': obs, 'OBGNME': ob_groups, 'WEIGHT': weights, 'ParSen_Weight' : weights})
        self._obs_data.set_index('OBSNME', inplace=True)
        
        if drop_regul is True:
            self.drop_regul(calc_sensitivity=False)
        if drop_groups is not None:
            self.drop_groups(drop_groups=drop_groups, calc_sensitivity=False)
        if keep_groups is not None:
            self.keep_groups(keep_groups = keep_groups, calc_sensitivity=False)
        if keep_obs is not None:
            self.keep_obs(keep_obs=keep_obs, calc_sensitivity=False)
        if remove_obs is not None:
            self.remove_obs(remove_obs=remove_obs, calc_sensitivity=False)
      
        # Fill DataFrame
        self.df = self.calc_sensitivity()

    def calc_sensitivity(self):
        # Get count of non-zero weights
        weights = self._obs_data['ParSen_Weight'].values
        n_nonzero_weights = np.count_nonzero(weights)

        # Calculate sensitivities
        # Could probally speed this up with some of the Matix class methods
        # - for another time..
        sensitivities = []
        for col in self.jco_df:
            sen = np.linalg.norm(np.asarray(self.jco_df[col])*weights)/n_nonzero_weights
            sensitivities.append(sen)

        # Build Group Array
        par_groups = []
        for par in self.jco_df.columns:
            par_group = self._pars_dict[par]
            par_groups.append(par_group)

        # Build pandas data frame of parameter sensitivities
        sen_data = {'Sensitivity': sensitivities, 'Parameter Group': par_groups}
        df = pd.DataFrame(sen_data, index=self.jco_df.columns)
        return df

    def drop_regul(self, calc_sensitivity = True):
        '''
        Recalculate sensitivity without regularization observations
        '''
        for index, row in self._obs_data.iterrows():    
            # Set weights for regularization info to zero
            if 'regul' in row['OBGNME'].lower():
                self._obs_data.set_value(index, 'ParSen_Weight', 0.0)
        if calc_sensitivity is True:
            self.df = self.calc_sensitivity()
        
    def drop_groups(self, drop_groups, calc_sensitivity = True):
        '''
        Recalculate sensitivity without groups
        '''
        for index, row in self._obs_data.iterrows():    
            # Set weights for obs in groups to zero
            if row['OBGNME'].lower() in drop_groups:
                self._obs_data.set_value(index, 'ParSen_Weight', 0.0)
        if calc_sensitivity is True:
            self.df = self.calc_sensitivity()

    def keep_groups(self, keep_groups, calc_sensitivity = True):
        '''
        Recalculate sensitivity with only groups
        '''
        for index, row in self._obs_data.iterrows():
            # Set weights for obs not in groups to zero
            if row['OBGNME'].lower() not in keep_groups:
                self._obs_data.set_value(index, 'ParSen_Weight', 0.0)
        if calc_sensitivity is True:
            self.df = self.calc_sensitivity()
        
    def keep_obs(self, keep_obs, calc_sensitivity = True):
        '''
        Recalculate sensitvity with only obs
        '''
        for index, row in self._obs_data.iterrows():
            # Set weights for obs not in keep_obs to zero
            if index.lower() not in keep_obs:
                self._obs_data.set_value(index, 'ParSen_Weight', 0.0)
        if calc_sensitivity is True:
            self.df = self.calc_sensitivity()
        
    def remove_obs(self, remove_obs, calc_sensitivity = True):
        '''
        Recalculate sensitivity without obs
        '''
        for index, row in self._obs_data.iterrows():
            # Set weights for obs in obs to zero
            if index.lower() in remove_obs:
                self._obs_data.set_value(index, 'ParSen_Weight', 0.0)
        if calc_sensitivity is True:
            self.df = self.calc_sensitivity()
           

    def tail(self, n_tail):
        ''' Get the lest sensitive parameters
        Parameters
        ----------
        n_tail: int
            Number of parameters to get

        Returns
        ---------
        pandas Series
            Series of n_tail least sensitive parameters

        '''
        return self.df.sort_values(by='Sensitivity', ascending=False)\
            .tail(n=n_tail)['Sensitivity']

    def head(self, n_head):
        ''' Get the most sensitive parameters
        Parameters
        ----------
        n_head: int
            Number of parameters to get

        Returns
        -------
        pandas Series
            Series of n_head most sensitive parameters
        '''
        return self.df.sort_values(by='Sensitivity', ascending=False)\
            .head(n=n_head)['Sensitivity']

    def par(self, parameter):
        '''Return the sensitivity of a single parameter

        Parameters
        ----------
        parameter: string

        Returns
        ---------
        float
            sensitivity of parameter

        '''
        return self.df.xs(parameter)['Sensitivity']

    def group(self, group, n=None):
        '''Return the sensitivities of a parameter group

        Parameters
        ----------
        group: string

        n: {None, int}, optional
            If None then return all parameters from group, else n is the number
            of parameters to return.
            If n is less than 0 then return the least sensitive parameters
            If n is greater than 0 then return the most sensitive parameters

        Returns
        -------
        Pandas DataFrame

        '''
        group = group.lower()
        if n is None:
            n_head = len(self.df.index)
        else:
            n_head = n

        if n_head > 0:
            sensitivity = self.df.sort_values(by='Sensitivity',
                                       ascending = False).ix[self.df['Parameter Group'] == group].head(n=n_head)
        if n_head < 0:
            n_head = abs(n_head)
            sensitivity = self.df.sort_values(by='Sensitivity',
                                       ascending = False).ix[self.df['Parameter Group'] == group].tail(n=n_head)

        sensitivity.index.name = 'Parameter'
        return sensitivity

    def sum_group(self):
        ''' Return sum of all parameters sensitivity by group

        Returns
        -------
        Pandas DataFrame
        '''
        sen_grouped = self.df.groupby(['Parameter Group'])\
            .aggregate(np.sum).sort_values(by='Sensitivity', ascending=False)
        return sen_grouped

    def plot(self, n=None, group=None, color_dict=None, alt_labels=None, **kwds):
        ''' Plot parameter sensitivities

        Parameters
        ----------
        n : int, optional
            Number of parameters to plot.  Default is None which will plot
            all parameters

        group : str, optional
            Parameter group to plot

        color_dict : dict, optional
            Dictionary where keys are the parameter group and values are the
            color to use for the bars for that group

        alt_labels : dict, optional
            Alternative names to use for labelling parameters.  Dictionary key is
            the parameter name as defined in the PEST control file, value is
            the name to use for the label

        Returns
        -------
        Matplotlib plot
            Bar plot of mean of sensitivity by parameter group
        '''
        if n is None:
            n_head = len(self.df.index)
        else:
            n_head = n

        if group is None:

            if n_head > 0:
                sensitivity = self.df.sort_values(by='Sensitivity',
                                           ascending=False).head(n=n_head)
            if n_head < 0:
                n_head = abs(n_head)
                sensitivity = self.df.sort_values(by='Sensitivity',
                                           ascending=False).tail(n=n_head)

        if group is not None:
            group = group.lower()
            if n_head > 0:
                sensitivity = self.df.sort_values(by='Sensitivity',
                                           ascending=False).ix[self.df['Parameter Group'] == group].head(n=n_head)         
            if n_head < 0:
                n_head = abs(n_head)
                sensitivity = self.df.sort_values(by='Sensitivity',
                                           ascending=False).ix[self.df['Parameter Group'] == group].tail(n=n_head)

        if 'ylabel' not in kwds:
            kwds['ylabel'] = 'Parameter'
        if 'xlabel' not in kwds:
            kwds['xlabel'] = 'Parameter Sensitivity'

        plot_obj = plots.BarPloth(sensitivity, values_col='Sensitivity',
                                  group_col='Parameter Group',
                                  color_dict=color_dict,
                                  alt_labels=alt_labels, **kwds)
        plot_obj.generate()
        plot_obj.draw()

        return plot_obj.fig, plot_obj.ax

    def plot_mean_group(self, alt_labels=None, **kwds):

        ''' Plot mean of all parameters sensitivity by group

        Returns
        -------
        Matplotlib plot
            Bar plot of mean of sensitivity by parameter group
        '''
        sen_grouped = self.df.groupby(['Parameter Group'])\
            .aggregate(np.mean).sort_values(by='Sensitivity', ascending=False)

        if 'ylabel' not in kwds:
            kwds['ylabel'] = 'Parameter Group'
        if 'xlabel' not in kwds:
            kwds['xlabel'] = 'Mean of Parameter Sensitivity'

        plot_obj = plots.BarPloth(sen_grouped, values_col='Sensitivity',
                                  alt_labels=alt_labels, **kwds)
        plot_obj.generate()
        plot_obj.draw()

        return plot_obj.fig, plot_obj.ax

    def plot_sum_group(self, alt_labels=None, **kwds):

        ''' Plot sum of all parameters sensitivity by group

        Returns
        -------
        Matplotlib plot
            Bar plot of sum of sensitivity by parameter group
        '''

        sen_grouped = self.df.groupby(['Parameter Group'])\
            .aggregate(np.sum).sort_values(by='Sensitivity', ascending=False)

        if 'ylabel' not in kwds:
            kwds['ylabel'] = 'Parameter Group'
        if 'xlabel' not in kwds:
            kwds['xlabel'] = 'Sum of Parameter Sensitivity'

        plot_obj = plots.BarPloth(sen_grouped, values_col='Sensitivity',
                                  alt_labels=alt_labels, **kwds)
        plot_obj.generate()
        plot_obj.draw()

        return plot_obj.fig, plot_obj.ax
            
