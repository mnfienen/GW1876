# -*- coding: utf-8 -*-
"""
Created on Fri Sep 04 12:45:11 2015

@author: egc
"""

import numpy as np
import pandas as pd
import os
import plots
from mat_handler import jco as Jco
from pst_handler import pst as Pst



class ObSen(object):

    def __init__(self, basename=None, parameter_data=None, res_df=None, 
                 jco_df=None):

        ''' Create ObSen class

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
            Otherwise the jco is read in every time ObSen class is initialized.
            
            
        res_df : DataFrame, optional
            Residual DataFrame used to define the weights to 
            calculate the observation sensitivity.  Providing a
            res_df offers some efficiencies if working interactively.
            If not provided it will look for basename+'.res'.  
            Weights are not taken from PEST control file
            (.pst) because regularization weights in PEST conrtrol file do
            not reflect the current weights.

        Attributes
        ----------
        df : Pandas DataFrame
            DataFrame of observation sensitivity.  Index entries of the DataFrame
            are the observation names.  

        Methods
        -------
        #plot()
        tail()
        head()
        #par()
        group()
        sum_group()
        #plot_sum_group()
        #plot_mean_group()



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
        self.res_df.set_index('name', drop=False, inplace = True)

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
        self._obs_data = pd.DataFrame({'OBSNME': obs, 'OBGNME': ob_groups, 'WEIGHT': weights, 'ObSen_Weight' : weights})
        self._obs_data.set_index('OBSNME', inplace=True)
             
        # Fill DataFrame
        self.df = self.calc_sensitivity()

    def calc_sensitivity(self):
        # Calculate sensitivities
        # Could probally speed this up with some of the Matix class methods
        # - for another time..
        sensitivities = []
        n_pars = self.jco_df.shape[1]
        for row in self.jco_df.iterrows():
            weight = self._obs_data.ix[row[0]]['ObSen_Weight']
            sen = (weight*(np.linalg.norm(row[1].values)))/n_pars
            sensitivities.append(sen)

        # Build Group Array
        ob_groups = []
        for ob in self.jco_df.index:
            ob_group = self._obs_data.ix[ob]['OBGNME']
            ob_groups.append(ob_group)
            
        # Build pandas data frame of parameter sensitivities
        sen_data = {'Sensitivity': sensitivities, 'Observation Group' : ob_groups}

        df = pd.DataFrame(sen_data, index=self.jco_df.index)
        return df
        
        
    def tail(self, n_tail):
        ''' Get the lest sensitive observations
        Parameters
        ----------
        n_tail: int
            Number of observations to get

        Returns
        ---------
        pandas Series
            Series of n_tail least sensitive observations

        '''
        return self.df.sort(columns='Sensitivity', ascending=False)\
            .tail(n=n_tail)['Sensitivity']

    def head(self, n_head):
        ''' Get the most sensitive observations
        Parameters
        ----------
        n_head: int
            Number of obsservations to get

        Returns
        -------
        pandas Series
            Series of n_head most sensitive obsservations
        '''
        return self.df.sort(columns='Sensitivity', ascending=False)\
            .head(n=n_head)['Sensitivity']

    def ob(self, observation):
        '''Return the sensitivity of a single obsservation

        Parameters
        ----------
        obsservation: string

        Returns
        ---------
        float
            sensitivity of obsservation

        '''
        return self.df.xs(observation)['Sensitivity']

    def group(self, group, n=None):
        '''Return the sensitivities of a observation group

        Parameters
        ----------
        group: string

        n: {None, int}, optional
            If None then return all parameters from group, else n is the number
            of observations to return.
            If n is less than 0 then return the least sensitive observations
            If n is greater than 0 then return the most sensitive observations

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
            sensitivity = self.df.sort(columns='Sensitivity',
                                       ascending = False).ix[self.df['Observation Group'] == group].head(n=n_head)
        if n_head < 0:
            n_head = abs(n_head)
            sensitivity = self.df.sort(columns='Sensitivity',
                                       ascending = False).ix[self.df['Observation Group'] == group].tail(n=n_head)

        sensitivity.index.name = 'Observation'
        return sensitivity

    def sum_group(self):
        ''' Return sum of all observation sensitivity by group

        Returns
        -------
        Pandas DataFrame
        '''
        sen_grouped = self.df.groupby(['Observation Group'])\
            .aggregate(np.sum).sort(columns='Sensitivity', ascending=False)
        return sen_grouped

    def plot(self, n=None, group=None, color_dict=None, alt_labels=None, **kwds):
        ''' Plot observation sensitivities

        Parameters
        ----------
        n : int, optional
            Number of observations to plot.  Default is None which will plot
            all observations

        group : str, optional
            Observation group to plot

        color_dict : dict, optional
            Dictionary where keys are the observation group and values are the
            color to use for the bars for that group

        alt_labels : dict, optional
            Alternative names to use for labelling observation.  Dictionary key is
            the observation name as defined in the PEST control file, value is
            the name to use for the label

        Returns
        -------
        Matplotlib plot
            Bar plot  of sensitivity of observations
        '''
        if n is None:
            n_head = len(self.df.index)
        else:
            n_head = n

        if group is None:

            if n_head > 0:
                sensitivity = self.df.sort(columns='Sensitivity',
                                           ascending=False).head(n=n_head)
            if n_head < 0:
                n_head = abs(n_head)
                sensitivity = self.df.sort(columns='Sensitivity',
                                           ascending=False).tail(n=n_head)

        if group is not None:
            group = group.lower()
            if n_head > 0:
                sensitivity = self.df.sort(columns='Sensitivity',
                                           ascending=False).ix[self.df['Observation Group'] == group].head(n=n_head)         
            if n_head < 0:
                n_head = abs(n_head)
                sensitivity = self.df.sort(columns='Sensitivity',
                                           ascending=False).ix[self.df['Observation Group'] == group].tail(n=n_head)

        if 'ylabel' not in kwds:
            kwds['ylabel'] = 'Observation'
        if 'xlabel' not in kwds:
            kwds['xlabel'] = 'Observation Sensitivity'

        plot_obj = plots.BarPloth(sensitivity, values_col='Sensitivity',
                                  group_col='Observation Group',
                                  color_dict=color_dict,
                                  alt_labels=alt_labels, **kwds)
        plot_obj.generate()
        plot_obj.draw()

        return plot_obj.fig, plot_obj.ax        
        

    