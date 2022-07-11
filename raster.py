#!/usr/bin/env python3


################################################################################
# -- Preprocessing and analysis of the simulation results
################################################################################

import numpy as np; import pylab as pl; import os, pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress
from imp import reload
import defaultParams; reload(defaultParams); from defaultParams import *
import searchParams; reload(searchParams); from searchParams import C_rng

def boxoff(ax):
    if isinstance(ax, list):
        for ax_ in ax:
            ax_.spines['top'].set_visible(False)
            ax_.spines['right'].set_visible(False)
    elif isinstance(ax, np.ndarray):
        for ax_ in ax.flatten():
            ax_.spines['top'].set_visible(False)
            ax_.spines['right'].set_visible(False)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
def to_square_plots(ax):

    if len(ax.shape)>1:
        for i in range(ax.shape[0]):            
            for j in range(ax.shape[1]):
                ratio = ax[i, j].get_data_ratio()
                ax[i, j].set_aspect(1.0/ratio)
    else:
        for i in range(ax.shape[0]):
            ratio = ax[i].get_data_ratio()
            ax[i].set_aspect(1.0/ratio)

class raster():
    
    '''
    Class for analyzing simulation data of an inhibition stabilized network.
    
    Parameters
    ----------
    
    fl_path : Path for the data file (these files are discriminated by
                                      inh. and exc. conductances)
    
    
    Attributes
    ----------
    
    Ntrials : Number of trials
    
    NE, NI      : Size of excitatory, inhibitory population
    
    st_tr_time, end_tr_time : Python arrays that contain the time point at which 
                              trials start and end.
    
    '''
    
    def __init__(self, fl_path, Be, Bi):
        fl_path = fl_path.format(Be, Bi)
        fl = open(fl_path, 'rb'); sim_res = pickle.load(fl); fl.close()
        
        self.Ntrials = sim_res['Ntrials']*C_rng.size
        self.num_models = C_rng.size
        self.NE = sim_res['NE']
        self.NI = sim_res['NI']
        self.N = sim_res['N']
        self.Be = Be
        self.Bi = Bi
        
        self.Ttrans = sim_res['Ttrans']
        self.Tstim  = sim_res['Tstim']
        self.Tblank = sim_res['Tblank']
        self.Texp   = self.Ttrans + self.Tstim + self.Tblank*2
        
        if not "NI_pv" in globals():
            self.w_etoe = sim_res['W_EtoE']
            self.w_etoi = sim_res['W_EtoI']
            self.w_itoe = sim_res['W_ItoE']        
            self.w_itoi = sim_res['W_ItoI']
            del sim_res['W_EtoE'], sim_res['W_EtoI'], sim_res['W_ItoE'], sim_res['W_ItoI']
        
        self.sim_res = sim_res
        
        if len(sim_res[nn_stim_rng[-1]][2]) == self.Ntrials:
            self.trial_type = 'MultipleSim'
        else:
            self.trial_type = 'SingleSim'
            
    def plot_by_trial(self, ax, n=100, interval=[9000, 12000]):
        sel_tr = np.random.choice(self.Ntrials+1)
        sel_ids = np.sort(np.random.choice(np.arange(1, self.N), n))
        for pert_idx, pert in enumerate(nn_stim_rng):
            ids = self.sim_res[pert][2][sel_tr]['senders']
            times = self.sim_res[pert][2][sel_tr]['times']
            ids = ids[(times>interval[0]) & (times<interval[1])]
            times = times[(times>interval[0]) & (times<interval[1])]
            all_ids = []
            all_times = []
            inh_idx = np.where(sel_ids > self.NE)[0][0]
            for idx, id_ in enumerate(sel_ids):
                all_ids.append((idx+1)*np.ones(np.sum(ids==id_)))
                all_times.append(times[ids==id_])
            all_ids = np.concatenate(all_ids)
            all_times = np.concatenate(all_times)
            ax[pert_idx].scatter(all_times[all_ids<=inh_idx],
                                 all_ids[all_ids<=inh_idx],
                                 marker='|', color='red')
            ax[pert_idx].scatter(all_times[all_ids>inh_idx],
                                 all_ids[all_ids>inh_idx],
                                 marker='|', color='blue')
            
    def create_fig_subdir(self, path, dir_name):
        
        dir_path = os.path.join(path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        
        return dir_path