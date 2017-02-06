# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 10:06:30 2015

@author: egc
"""

import datetime
import numpy as np
import pandas as pd
import os


class Rmr(object):
    def __init__(self, basename=None):
        ''' Create RMR class
        Parameters
        ----------
        basename : str
            Basename for the PEST control file, if full path provided the current
            working directory is assumed
            
        Attributes
        ----------
        node_list : list
            List of nodes used
        
        data : list
           List of lists. Each node in node_list contains a separate list of 
           runtimes.  Index values from node_list correspond with index values 
           of data.
           
        node_average : list
          List of tuples.  Within each tuple index 1 is the node and index 2 
          is the average runtime in seconds
          
        Notes
        ------
        Currently only tested with BeoPEST.  PEST may have a different format
        for printing date-time to the .rmr file.
           
        '''
        if basename is not None:
            self.basename = os.path.split(basename)[-1].split('.')[0]
            self.directory = os.path.split(basename)[0]
            if len(self.directory) == 0:
                self.directory = os.getcwd()
                       
        self.rmr_file = os.path.join(self.directory, self.basename + '.rmr')
        rmr = open(self.rmr_file)    
        node_index = dict()
        run_starts = dict()
        run_stats = dict()
               
        for line in rmr:
            # Update node_index if necessary
            if "index of" in line:
                node = int(line.split('index of')[1].strip().split(' ')[0])
                directory = line.split('at working directory')[1].strip().split('"')[1]
                node_index[node] = directory
            if "commencing on node" in line:
                time = line.strip().split(':-')[0]
                # if seconds are 60 change to 59 then add second
                if time.split(':')[-1].split('.')[0] == '60':
                    time = time.split(':')[0]+':'+time.split(':')[1]+':59.00'
                    time = datetime.datetime.strptime(time, '%d %b %H:%M:%S.%f')
                    time = time + datetime.timedelta(seconds=1)
                else:
                    time = datetime.datetime.strptime(time, '%d %b %H:%M:%S.%f')
                time = time.replace(year = datetime.datetime.now().year)
                node = int(line.strip().split('commencing on node ')[1].strip().split('.')[0])
                run_starts[node_index[node]] = time
            if "completed on node" in line:
                time = line.replace('; old run so results not needed.','').strip().split(':-')[0]
                # if seconds are 60 change to 59 then add second
                if time.split(':')[-1].split('.')[0] == '60':
                    time = time.split(':')[0]+':'+time.split(':')[1]+':59.00'
                    time = datetime.datetime.strptime(time, '%d %b %H:%M:%S.%f')
                    time = time + datetime.timedelta(seconds=1)
                else:
                    time = datetime.datetime.strptime(time, '%d %b %H:%M:%S.%f')
                time = time.replace(year = datetime.datetime.now().year)
                node = int(line.replace('; old run so results not needed.','').strip().split('completed on node ')[1].strip(' .'))
                start = run_starts[node_index[node]]
                length_seconds = (time - start).total_seconds()
                if node_index[node] in run_stats:        
                    run_stats[node_index[node]].append(length_seconds)
                else:
                    run_stats[node_index[node]] = [length_seconds,]           
        
        self._run_stats = run_stats        
        self._node_index = node_index
        self._run_starts = run_starts
        # Process Run Stats     
        self._node_list = []
        for node in run_stats:
            self._node_list.append(node)
        self._node_list.sort()
        self.nodes = pd.DataFrame(self._node_list)
        self.nodes.columns = ['Node']
        
        self.data = []
        self.node_average = []
        for node in self._node_list:
            self.data.append(run_stats[node])
            
            average = np.array(run_stats[node]).mean()
            self.node_average.append((node, average))
        self.node_average = pd.DataFrame(self.node_average)
        self.node_average.columns = ['Node', 'Average Runtime']

# Need to move this to the plots class as some point            
    def boxplot(self):
        import matplotlib.pyplot as plt
        ''' Create a simple boxplot displaying runtime data for each node
        
        Returns
        -------
        Matplotlib boxplot
        '''        
        plt.boxplot(self.data)
        tick_locs, tick_labels = plt.xticks() 
        plt.xticks(tick_locs, self._node_list, rotation = 90, fontsize = 'x-small')
        plt.ylabel('Run Time (seconds)')
        plt.grid(True)         
        plt.tight_layout() 