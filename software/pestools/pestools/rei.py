__author__ = 'aleaf'

import os
import numpy as np
import pandas as pd
from .res import Res
from .pest import Pest
from matplotlib.backends.backend_pdf import PdfPages


class Rei(object):
    """
    Rei Class

    Parameters
    ----------
    basename : str
        basename for pest run (including path)

    obs_info_file : str, optional
        csv file containing observation locations and/or observation type.

    name_col : str, default 'Name'
        column in obs_info_file containing observation names

    x_col : str, default 'X'
        column in obs_info_file containing observation x locations

    y_col : str, default 'Y'
        column in obs_info_file containing observation y locations

    type_col : str, default 'Type'
        column in obs_info_file containing observation types (e.g. heads, fluxes, etc). A single
        type ('observation') is assigned in the absence of type information

    Attributes
    ----------
    df : DataFrame
        contains all of the information from the res or rei file; is used to build phi dataframe

    phi_by_group : DataFrame
        contains phi contribution by group for each iteration

    phi_by_type : DataFrame
        contains phi contribution by observation type for each iteration

    phi_by_component : DataFrame
        contains phi contribution by objective function component for each iteration

    obsinfo : DataFrame
        contains information from observation information file, and also observation groups

    Notes
    ------
    Column names in the observation information file are remapped to their default values after import



    Attributes
    ----------
    reifiles : dict
        Dictionary of rei files (strings) produced from PEST run
        with residuals information saved every iteration (REISAVEITN).
        Integer keys represent iteration number.

    Methods
    -------
    one2one_plots()


    Notes
    ------

    """

    def __init__(self, basename, obs_info_file=None, name_col='Name',
                 x_col='X', y_col='Y', type_col='Type',
                 basename_col='basename', datetime_col='datetime', group_cols=[],
                 **kwds):

        #Pest.__init__(self, basename, obs_info_file=obs_info_file)
        self.basename = basename
        self._Pest = Pest(basename, obs_info_file=obs_info_file)
        self.obsinfo = self._Pest.obsinfo
        self.obs_groups = self._Pest.obs_groups
        self._obstypes = pd.DataFrame({'Type': ['observation'] * len(self.obs_groups)}, index=self.obs_groups)

        #self.phi = self.obsinfo.copy()
        self.phi = pd.DataFrame()
        self.phi_by_group = pd.DataFrame(columns=self.obs_groups)
        self.phi_by_type = pd.DataFrame()
        self.phi_by_component = pd.DataFrame()

        # list rei files for run
        reifiles = [f for f in os.listdir(os.path.split(basename)[0])
                    if os.path.split(basename)[1] + '.rei' in f]

        # sort by iteration number (may not be the most elegant approach)
        self.reifiles = {}
        for f in reifiles:
            try:
                i = int(f.split('.')[-1])
                self.reifiles[i] = os.path.join(self.run_folder, f)
            except:
                continue
        # for SVDA runs, may not have .0 (initial) rei file. Get rei file for base run.
        if 0 not in list(self.reifiles.keys()):
            self._read_svda()
            self.reifiles[0] = os.path.join(self.run_folder, self.BASEPESTFILE[:-4] + '.rei')

    def plot_one2ones(self, groupinfo, outpdf='', **kwds):

        if len(outpdf) == 0:
            outpdf = self.basename + '_reis.pdf'

        print('plotting...')
        pdf = PdfPages(outpdf)
        for i in self.reifiles.keys():
            print('{}'.format(self.reifiles[i]))
            r = Res(self.reifiles[i])
            fig, ax = r.plot_one2one(groupinfo, title='Iteration {}'.format(i), **kwds)

            pdf.savefig(fig, **kwds)
        print('\nsaved to {}'.format(outpdf))
        pdf.close()

    def get_phi(self):
        print('getting phi by group for each iteration...')
        for i in self.reifiles.keys():
            print('{}'.format(self.reifiles[i]))
            r = Res(self.reifiles[i])
            phi = r.phi.Weighted_Sq_Residual
            #phi.name = i
            self.phi[i] = phi
            #self.phi = self.phi.append(phi.T)
            #self.phi_by_group = self.phi_by_group.append(phi.T)
            #self.phi_by_group.index.name = 'Pest iteration'

        # get phi just for observation groups
        self.phi_obs_by_group = self.phi_by_group.ix[:, self.obsgroups]

        # get phi by observation type for each iteration
        for type in np.unique(self._obstypes.Type):
            typegroups = self._obstypes[self._obstypes.Type == type].index.tolist()
            self.phi_by_type[type] = self.phi_by_group.ix[:, typegroups].sum(axis=1)
            self.phi_by_type.index.name = 'Pest iteration'

        # get phi by component for each iteration
        self.phi_by_component['Measurement Phi'] = self.phi_obs_by_group.sum(axis=1)
        if len(self.reggroups) > 0:
            self.phi_by_component['Regularisation Phi'] = self.phi_by_group.ix[:, self.reggroups].sum(axis=1)
        self.phi_by_component['Phi Total'] = self.phi_by_component.sum(axis=1)
        self.phi_by_component.index.name = 'Pest iteration'





