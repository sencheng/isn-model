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

class simdata():
    
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
        self.Nall = sim_res['N']
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
        
        self.all_fr_exc = np.array([])#Storing concatenated firing rates
        self.all_fr_inh = np.array([])#Storing concatenated firing rates
        self.all_fr_time = np.array([])#Storing concatenated firing rates
        
        self.trans_ticks = np.zeros_like(nn_stim_rng)
        self.base_ticks = np.zeros_like(nn_stim_rng)
        self.stim_ticks = np.zeros_like(nn_stim_rng)
        
        self.trans_cv = np.zeros((2, self.Ntrials))
        self.base_cv  = np.zeros_like(self.trans_cv)
        self.stim_cv  = np.zeros_like(self.trans_cv)
        
        self.trans_ff = np.zeros_like(self.trans_cv)
        self.base_ff  = np.zeros_like(self.trans_cv)
        self.stim_ff  = np.zeros_like(self.trans_cv)
        
        self.data_violin = []
        self.data_violin_mean = []
        
        self.max_diff = -np.inf
        
    def get_trial_times(self):
        
        '''
        Trials appear in a single simulation session one after another. This
        method extract starting (attr: st_tr_time) and finishing time
        (attr: end_tr_time) of each trial.
        '''
        
        self.st_tr_time = np.arange(self.Ntrials)*self.Texp
        self.end_tr_time = np.arange(1, self.Ntrials+1)*self.Texp
        
    def get_ind_cond(self, Id, g_in, g_ex):
        
        """
        Calculates the inhibitory input conductances
        of individual neurons for a given duration.
        
        Inputs
        ------
        Id : numpy array
             containing neuron ids for the given duration.
        
        duration :  int
                    The time interval during which the recording is performed.
        
        is_ms : bool
                whether the duration is in ms or not. Default is True.
        
        Outputs
        -------
        
        cond_e, cond_i : (numpy array, numpy array)
                         inhibitory input conductance of individual excitatory
                         and inhibitory neurons.
        """
        ids = np.unique(Id)
        avg_curr_i = np.zeros(ids.shape)
        avg_curr_e = np.zeros(ids.shape)
        
        for i, ID in enumerate(ids):
            avg_curr_i[i] = g_in[Id==ID].mean()
            avg_curr_e[i] = g_ex[Id==ID].mean()
            
        # curr_e = avg_curr[ids<=NE]
        # curr_i = avg_curr[ids>NE]
        
        return avg_curr_e, avg_curr_i
            
        
    def get_ind_fr(self, Id, duration, is_ms=True):
        
        """
        Calculates the firing rate of individual neurons for a given duration.
        
        Inputs
        ------
        Id : numpy array
             containing neuron ids for the given duration.
        
        duration :  int
                    The time interval during which the recording is performed.
        
        is_ms : bool
                whether the duration is in ms or not. Default is True.
        
        Outputs
        -------
        
        fr_e, fr_i : (numpy array, numpy array)
                    Average firing rate of individual excitatory
                    and inhibitory neurons.
        """
        
        if is_ms:
            c_factor = 1000
        else:
            c_factor = 1
            
        fr_e = np.histogram(Id[Id <= self.NE], np.arange(1, self.NE+1.1))[0]/duration*c_factor
        fr_i = np.histogram(Id[Id > self.NE], np.arange(self.NE+1, self.Nall+1.1))[0]/duration*c_factor

        return fr_e, fr_i
    
    def smooth(self, signal, win_size=20, kernel='rect'):
        
        '''
        Smoothen the input signal.
        
        Inputs
        ------
        
        signal : The signal to be smoothened.
        
        win_size : Smoothing filter size. Default is 20.
        
        kernel : The flitering kernel. The only available option is "rect"
        '''
        
        if kernel=='rect':
            
            filt = np.ones(win_size)/win_size
            
        # else:
            
        #     NotImplementedError
        
        return np.convolve(signal, filt, 'same')
    
    def get_pop_fr(self, Id, Times, edges,
                   is_ms=True, with_silents=True, smooth=True):
        
        if is_ms:
            c_factor = 1000
        else:
            c_factor = 1
            
        exc_ids = Id[Id<=self.NE]
        inh_ids = Id[Id>self.NE]
        
        exc_times = Times[Id<=self.NE]
        inh_times = Times[Id>self.NE]
            
        if not with_silents:
            
            exc_num = np.unique(exc_ids).size
            inh_num = np.unique(inh_ids).size
            
        else:
            
            exc_num = self.NE
            inh_num = self.NI
            
        # edges = np.arange(st_time, st_time+self.Tstim+self.Ttrans+self.Tblank)
        
        fr_e = np.histogram(exc_times, edges)[0]/exc_num*c_factor/np.diff(edges)[0]
        fr_i = np.histogram(inh_times, edges)[0]/inh_num*c_factor/np.diff(edges)[0]
        
        fr_e = self.smooth(fr_e, win_size=10)
        fr_i = self.smooth(fr_i, win_size=10)
        
        return fr_e, fr_i
    
    def get_ind_cv(self, ID, T):
        
        u_id = np.unique(ID)
        cv = []
        
        for i, Id in enumerate(u_id):
            
            spks = T[ID==Id]
            spks.sort()
            if spks.size > 2:            
                ISI = np.diff(spks)
                cv.append(ISI.var()/(ISI.mean()**2))
                             
        return cv
    
    def get_cv(self, pert_val, interval, ff_bin=2):
        
        if not hasattr(self, 'st_tr_time'):
            self.get_trial_times()
        
        if self.trial_type == 'SingleSim':
            spk_times = self.sim_res[pert_val][2]['times']
            spk_ids   = self.sim_res[pert_val][2]['senders']
        
        e_cv = np.zeros(self.Ntrials)
        i_cv, e_ff, i_ff = np.zeros_like(e_cv), np.zeros_like(e_cv), np.zeros_like(e_cv)
        
        for tr in range(self.Ntrials):
            
            # sel_sp_t = spk_times[(spk_times >= self.st_tr_time[i]+interval[0]) & 
            #                      (spk_times <  self.st_tr_time[i]+interval[1])]
            
            if self.trial_type == 'SingleSim':
            
                sel_id   = spk_ids[(spk_times >= self.st_tr_time[tr]+interval[0]) &
                                   (spk_times <  self.st_tr_time[tr]+interval[1])]
                
                spk_time = spk_times[(spk_times >= self.st_tr_time[tr]+interval[0]) &
                                     (spk_times <  self.st_tr_time[tr]+interval[1])]
                
            else:
                
                sel_id = self.sim_res[pert_val][2][tr]['times']
                spk_time = self.sim_res[pert_val][2][tr]['senders']
            
            e_id = sel_id[sel_id<=self.NE]
            e_t  = spk_time[sel_id<=self.NE]
            i_id = sel_id[sel_id> self.NE]
            i_t  = spk_time[sel_id>self.NE]
            
            e_cv[tr] = np.mean(self.get_ind_cv(e_id, e_t))
            i_cv[tr] = np.mean(self.get_ind_cv(i_id, i_t))
            
            e_fr_bin = np.histogram(e_t, np.arange(self.st_tr_time[tr]+interval[0],
                                                   self.st_tr_time[tr]+interval[1],
                                                   ff_bin))[0]/self.NE/np.diff(interval)*1000
            e_ff[tr] = e_fr_bin.var()/e_fr_bin.mean()
            
            i_fr_bin = np.histogram(i_t, np.arange(self.st_tr_time[tr]+interval[0],
                                                   self.st_tr_time[tr]+interval[1],
                                                   ff_bin))[0]/self.NI/np.diff(interval)*1000
            i_ff[tr] = e_fr_bin.var()/e_fr_bin.mean()
            
        return e_cv, i_cv, e_ff, i_ff
    
    def get_cond(self, pert_val, interval):
        
        if not hasattr(self, 'st_tr_time'):
            self.get_trial_times()
        
        cond_inh = np.zeros((self.Nall, self.Ntrials))
        cond_exc = np.zeros((self.Nall, self.Ntrials))
        
        if self.trial_type == 'SingleSim':
            times = self.sim_res[pert_val][3]['times']
            ids   = self.sim_res[pert_val][3]['senders']
            conds_i = self.sim_res[pert_val][3]['g_in']
            conds_e = self.sim_res[pert_val][3]['g_ex']
        
        for tr in range(self.Ntrials):
            
            # sel_sp_t = spk_times[(spk_times >= self.st_tr_time[i]+interval[0]) & 
            #                      (spk_times <  self.st_tr_time[i]+interval[1])]
            
            if self.trial_type == 'SingleSim':
                sel_id = ids[(times >= self.st_tr_time[tr]+interval[0]) &
                             (times <  self.st_tr_time[tr]+interval[1])]
                
                sel_g_i  = conds_i[(times >= self.st_tr_time[tr]+interval[0]) &
                                   (times <  self.st_tr_time[tr]+interval[1])]
                
                sel_g_e  = conds_e[(times >= self.st_tr_time[tr]+interval[0]) &
                                   (times <  self.st_tr_time[tr]+interval[1])]
                
            else:
                times = self.sim_res[pert_val][3][tr]['times']
                ids   = self.sim_res[pert_val][3][tr]['senders']
                conds_i = self.sim_res[pert_val][3][tr]['g_in']
                conds_e = self.sim_res[pert_val][3][tr]['g_ex']
                
                sel_id = ids[(times >= interval[0]) &
                             (times <  interval[1])]
                
                sel_g_i  = conds_i[(times >= interval[0]) &
                                   (times <  interval[1])]
                
                sel_g_e  = conds_e[(times >= interval[0]) &
                                   (times <  interval[1])]
                
            cond_exc[:, tr], cond_inh[:, tr] = self.get_ind_cond(sel_id,
                                                                 sel_g_i,
                                                                 sel_g_e)
            
        return cond_exc, cond_inh
    
    def get_fr_surrogate(self, pert_val, interval, NUM_SHUF=1000):
        
        if not hasattr(self, 'st_tr_time'):
            self.get_trial_times()
        
        fr_inh = np.zeros((self.NI, NUM_SHUF))
        fr_exc = np.zeros((self.NE, NUM_SHUF))
        
        num_spks_e = self.base_exc.sum(axis=1)*self.Tblank/1000. + \
                     self.stim_exc.sum(axis=1)*self.Tstim/1000.
        num_spks_i = self.base_inh.sum(axis=1)*self.Tblank/1000. + \
                     self.stim_inh.sum(axis=1)*self.Tstim/1000.
        
        for i in range(num_spks_e.shape[0]):
            spk_times_e = np.random.uniform(low=interval[0],
                                            high=interval[1],
                                            size=(int(num_spks_e[i]), NUM_SHUF))
            
            for j in range(NUM_SHUF):
                spks = spk_times_e[:, j]
                base = (spks < interval[0] + self.Tblank).sum()/self.Tblank/self.Ntrials
                stim = (spks >= interval[0] + self.Tblank).sum()/self.Tstim/self.Ntrials
                fr_exc[i, j] = stim-base
                
        for i in range(num_spks_i.shape[0]):
            spk_times_i = np.random.uniform(low=interval[0],
                                            high=interval[1],
                                            size=(int(num_spks_i[i]), NUM_SHUF))
            for j in range(NUM_SHUF):
                spks = spk_times_i[:, j]
                base = (spks < interval[0] + self.Tblank).sum()/self.Tblank/self.Ntrials
                stim = (spks >= interval[0] + self.Tblank).sum()/self.Tstim/self.Ntrials
                fr_inh[i, j] = stim-base
            
        return fr_exc*1000, fr_inh*1000
        
    def get_fr(self, pert_val, interval):
        
        if not hasattr(self, 'st_tr_time'):
            self.get_trial_times()
        
        fr_inh = np.zeros((self.NI, self.Ntrials))
        fr_exc = np.zeros((self.NE, self.Ntrials))
        
        if self.trial_type == 'SingleSim':
        
            spk_times = self.sim_res[pert_val][2]['times']
            spk_ids   = self.sim_res[pert_val][2]['senders']
        
        for tr in range(self.Ntrials):
            
            # sel_sp_t = spk_times[(spk_times >= self.st_tr_time[i]+interval[0]) & 
            #                      (spk_times <  self.st_tr_time[i]+interval[1])]
            if self.trial_type == 'SingleSim':
                sel_id   = spk_ids[(spk_times >= self.st_tr_time[tr]+interval[0]) &
                                   (spk_times <  self.st_tr_time[tr]+interval[1])]
            else:
                spk_ids = self.sim_res[pert_val][2][tr]['senders']
                spk_times = self.sim_res[pert_val][2][tr]['times']
                sel_id   = spk_ids[(spk_times >= interval[0]) &
                                   (spk_times <  interval[1])]
                
            
            e, i = self.get_ind_fr(sel_id, np.diff(interval), is_ms=True)   
            
            fr_exc[:, tr] = e
            fr_inh[:, tr] = i
            
        return fr_exc, fr_inh
    
    def get_fr_time(self, pert_val, interval):
        
        if not hasattr(self, 'st_tr_time'):
            self.get_trial_times()
            
        bin_size = 10
        T_edges_def = np.arange(interval[0],
                                interval[1],
                                bin_size)
        
        fr_inh = np.zeros((self.Ntrials, T_edges_def.size-1))
        fr_exc = np.zeros((self.Ntrials, T_edges_def.size-1))
        
        if self.trial_type == 'SingleSim':
            spk_times = self.sim_res[pert_val][2]['times']
            spk_ids   = self.sim_res[pert_val][2]['senders']
        
        for tr in range(self.Ntrials):
            
            if self.trial_type == 'SingleSim':
            
                T_edges = T_edges_def + self.st_tr_time[tr]
                # sel_sp_t = spk_times[(spk_times >= self.st_tr_time[i]+interval[0]) & 
                #                      (spk_times <  self.st_tr_time[i]+interval[1])]
                sel_id   = spk_ids[(spk_times >= self.st_tr_time[tr]+interval[0]) &
                                   (spk_times <  self.st_tr_time[tr]+interval[1])]
                sel_T    = spk_times[(spk_times >= self.st_tr_time[tr]+interval[0]) &
                                     (spk_times <  self.st_tr_time[tr]+interval[1])]
                
            else:
            
                sel_id = self.sim_res[pert_val][2][tr]['senders']
                sel_T  = self.sim_res[pert_val][2][tr]['times']
                T_edges = T_edges_def
            
            e, i = self.get_pop_fr(sel_id, sel_T, T_edges, is_ms=True)
            print(tr, e.shape)
            
            fr_exc[tr, :] = e
            fr_inh[tr, :] = i
            
        return fr_exc, fr_inh, T_edges[0:-1]+bin_size/2
    
    def get_cond_diff(self, pert_val):
        
        print("Computing conductance changes for pert={}".format(pert_val))
        
        '''
        This method calculates the change of average firing rate for each
        individual neuron in each trial due to perturbation.
        
        Parameters
        ----------
        pert_val : perturbation value
        '''
        
        self.base_e_cond, self.base_i_cond = self.get_cond(pert_val,
                                                           [self.Ttrans,
                                                            self.Ttrans+self.Tblank])
        self.stim_e_cond, self.stim_i_cond = self.get_cond(pert_val,
                                                           [self.Ttrans+self.Tblank, 
                                                            self.Ttrans+self.Tblank+self.Tstim])
            
        self.diff_e_cond = self.stim_e_cond - self.base_e_cond
        self.diff_i_cond = self.stim_i_cond - self.base_i_cond
        
        self.diff_cond = {"E": {"ge": self.diff_e_cond[0:NE, :],
                                "gi": self.diff_i_cond[0:NE, :]},
                          "I": {"ge": self.diff_e_cond[NE:, :],
                                "gi": self.diff_i_cond[NE:, :]}}
        
        self.diff_exc_m = self.diff_e_cond.mean(axis=1)
        self.diff_inh_m = self.diff_i_cond.mean(axis=1)
        
    def get_fr_diff_mean(self):
        
        Ntrials_per_model = Ntrials
        self.diff_exc_m = np.zeros((self.diff_exc.shape[0], self.num_models))
        self.diff_inh_m = np.zeros((self.diff_inh.shape[0], self.num_models))
        self.diff_exc_m_include_per = np.zeros((self.diff_exc.shape[0], self.num_models))
        self.diff_inh_m_include_per = np.zeros((self.diff_exc.shape[0], self.num_models))
        
        for i in range(self.base_exc.shape[0]):
            for mdl_i in range(self.num_models):
                base_exc = self.base_exc[i][mdl_i*Ntrials_per_model:(mdl_i+1)*Ntrials_per_model]
                stim_exc = self.stim_exc[i][mdl_i*Ntrials_per_model:(mdl_i+1)*Ntrials_per_model]
                diff_exc = self.diff_exc[i][mdl_i*Ntrials_per_model:(mdl_i+1)*Ntrials_per_model]
                IND = (base_exc!=0) | (stim_exc!=0)
                self.diff_exc_m[i, mdl_i] = np.mean(diff_exc[IND])
                self.diff_exc_m_include_per[i, mdl_i] = IND.mean()*100
            
        for i in range(self.base_inh.shape[0]):
            for mdl_i in range(self.num_models):
                base_inh = self.base_inh[i][mdl_i*Ntrials_per_model:(mdl_i+1)*Ntrials_per_model]
                stim_inh = self.stim_inh[i][mdl_i*Ntrials_per_model:(mdl_i+1)*Ntrials_per_model]
                diff_inh = self.diff_inh[i][mdl_i*Ntrials_per_model:(mdl_i+1)*Ntrials_per_model]
                IND = (base_inh!=0) | (stim_inh!=0)
                self.diff_inh_m[i, mdl_i] = np.mean(diff_inh[IND])
                self.diff_inh_m_include_per[i, mdl_i] = IND.mean()*100        
            
        
        
    def get_fr_diff(self, pert_val, exclude_inactives=True, significance_test=False):
        
        print("Computing firing rate changes for pert={}".format(pert_val))
        
        '''
        This method calculates the change of average firing rate for each
        individual neuron in each trial due to perturbation.
        
        Parameters
        ----------
        pert_val : perturbation value
        '''
        
        self.base_exc, self.base_inh = self.get_fr(pert_val,
                                                   [self.Ttrans,
                                                    self.Ttrans+self.Tblank])
        self.stim_exc, self.stim_inh = self.get_fr(pert_val,
                                                   [self.Ttrans+self.Tblank, 
                                                    self.Ttrans+self.Tblank+self.Tstim])
        
        if significance_test:
            self.diff_exc_shuf, self.diff_inh_shuf = \
                self.get_fr_surrogate(pert_val,
                                      [self.Ttrans, 
                                      self.Ttrans+self.Tblank+self.Tstim])
        '''
        self.trans_cv[0, :], self.trans_cv[1, :], self.trans_ff[0, :], self.trans_ff[1, :] = \
        self.get_cv(pert_val, [0, self.Ttrans])
        
        self.base_cv[0, :], self.base_cv[1, :], self.base_ff[0, :], self.base_ff[1, :] = \
        self.get_cv(pert_val, [self.Ttrans, self.Ttrans+self.Tblank])
        
        self.stim_cv[0, :], self.stim_cv[1, :], self.stim_ff[0, :], self.stim_ff[1, :] = \
        self.get_cv(pert_val, [self.Ttrans+self.Tblank, self.Ttrans+self.Tblank+self.Tstim])
        '''
        self.diff_exc = self.stim_exc - self.base_exc
        self.diff_inh = self.stim_inh - self.base_inh
        
        self.get_ids_of_changing_units()
        
        if 'NI_pv' in globals():
            
            self.base_inh_pv = self.base_inh[:NI_pv]
            self.stim_inh_pv = self.stim_inh[:NI_pv]
            
            self.base_inh_som = self.base_inh[NI_pv:NI-NI_vip]
            self.stim_inh_som = self.stim_inh[NI_pv:NI-NI_vip]
            
            self.base_inh_vip = self.base_inh[NI-NI_vip:]
            self.stim_inh_vip = self.stim_inh[NI-NI_vip:]
            
            self.diff_inh_pv = self.diff_inh[:NI_pv]
            self.diff_inh_som = self.diff_inh[NI_pv:NI-NI_vip]
            self.diff_inh_vip = self.diff_inh[NI-NI_vip:]
        
        
        self.get_fr_diff_mean()
        
        
        
        # self.diff_exc_m = self.diff_exc.mean(axis=1)
        # self.diff_inh_m = self.diff_inh.mean(axis=1)
        
        if exclude_inactives:
            
            self.diff_exc = self.diff_exc[(self.base_exc!=0) | (self.stim_exc!=0)]
            self.diff_inh = self.diff_inh[(self.base_inh!=0) | (self.stim_inh!=0)]
            
            self.base_inh_nz = self.base_inh[(self.base_inh!=0) | (self.stim_inh!=0)]
            self.base_exc_nz = self.base_exc[(self.base_exc!=0) | (self.stim_exc!=0)]
            
            if hasattr(self, 'diff_inh_pv'):
                
                self.diff_inh_pv = self.diff_inh_pv[(self.base_inh_pv!=0) |
                                                    (self.stim_inh_pv!=0)]
                
                self.diff_inh_som = self.diff_inh_som[(self.base_inh_som!=0) |
                                                      (self.stim_inh_som!=0)]
                
                self.diff_inh_vip = self.diff_inh_vip[(self.base_inh_vip!=0) |
                                                      (self.stim_inh_vip!=0)]
            
            #self.paradox_score = ((np.sum(self.diff_inh>0)/self.diff_inh.size -\
            #                       np.sum(self.diff_exc>0)/self.diff_exc.size)*\
            #                       np.sum(self.diff_inh>0)/self.diff_inh.size)

            # self.paradox_score = np.sign(np.sum(self.diff_inh>0)/self.diff_inh.size-0.5)*\
            #                              (np.sum(self.diff_inh>0)/self.diff_inh.size/\
            #                              (np.sum(self.diff_exc>0)/self.diff_exc.size))
                                             
            if np.sum(self.diff_inh>0)/self.diff_inh.size > 0.5:
                self.paradox_score = np.sum(self.diff_inh>0)/self.diff_inh.size-\
                                     np.sum(self.diff_exc>0)/self.diff_exc.size
            else:
                self.paradox_score = np.nan

            # self.diff_exc_m = self.diff_exc.mean()
            # self.diff_inh_m = self.diff_inh.mean()
            
    def get_significant_changes(self, pval=0.05):
        
        sig_dec_inh, sig_inc_inh, non_sig_inh = [0]*3        
        sig_dec_exc, sig_inc_exc, non_sig_exc = [0]*3
        
        for i, v in enumerate(self.diff_inh_m):
            p_val_dec = np.sum(self.diff_inh_shuf[i, :]<v)/self.diff_inh_shuf.shape[1]
            p_val_inc = np.sum(self.diff_inh_shuf[i, :]>v)/self.diff_inh_shuf.shape[1]
            
            if (p_val_dec < pval) & (p_val_inc < pval):
                print("strange!")
            else:
                if p_val_dec < pval:
                    sig_dec_inh += 1
                elif p_val_inc < pval:
                    sig_inc_inh += 1
                else:
                    non_sig_inh += 1
                
        for i, v in enumerate(self.diff_exc_m):
            p_val_dec = np.sum(self.diff_exc_shuf[i, :]<v)/self.diff_exc_shuf.shape[1]
            p_val_inc = np.sum(self.diff_exc_shuf[i, :]>v)/self.diff_exc_shuf.shape[1]
            
            if (p_val_dec < pval) & (p_val_inc < pval):
                print("strange!")
            else:
                if p_val_dec < pval:
                    sig_dec_exc += 1
                elif p_val_inc < pval:
                    sig_inc_exc += 1
                else:
                    non_sig_exc += 1
        INH = [sig_dec_inh, sig_inc_inh, non_sig_inh] 
        EXC = [sig_dec_exc, sig_inc_exc, non_sig_exc]
        return EXC, INH
        
    def get_avg_frs(self, pert_val):
        
        self.trans_fr = self.get_fr_time(pert_val,
                                         [0,
                                         self.Ttrans])
        
        self.base_fr = self.get_fr_time(pert_val,
                                        [self.Ttrans,
                                        self.Ttrans+self.Tblank])
        
        self.stim_fr = self.get_fr_time(pert_val,
                                        [self.Ttrans+self.Tblank,
                                        self.Ttrans+self.Tblank+self.Tstim])
        
        self.all_fr = [np.hstack((self.trans_fr[0], self.base_fr[0], self.stim_fr[0])), 
                       np.hstack((self.trans_fr[1], self.base_fr[1], self.stim_fr[1])),
                       np.concatenate((self.trans_fr[2], self.base_fr[2], self.stim_fr[2]))]
        
    def concat_avg_frs_perts(self, pert_id):
        
        self.all_fr_exc = np.concatenate((self.all_fr_exc,
                                          self.all_fr[0].mean(axis=0)))
        
        self.all_fr_inh = np.concatenate((self.all_fr_inh,
                                          self.all_fr[1].mean(axis=0)))
        
        time_vec = self.all_fr[2] - self.all_fr[2].min() + \
                   pert_id*self.Texp
        
        self.trans_ticks[pert_id] = pert_id*self.Texp
        self.base_ticks[pert_id]  = self.Ttrans + pert_id*self.Texp
        self.stim_ticks[pert_id]  = self.Ttrans + self.Tblank + pert_id*self.Texp
        
        self.all_fr_time = np.concatenate((self.all_fr_time,
                                           time_vec))
        
    def plot_avg_frs(self, ax):
        
        ax.plot(self.all_fr_time, self.all_fr_exc, color='red', label='exc')
                
        ax.plot(self.all_fr_time, self.all_fr_inh, color='blue', label='inh')
        
        ylim = ax.get_ylim()
            
    # def add_indicator_lines(self, ax):
        
        D = np.diff(ylim)/10
        
        for i in range(self.trans_ticks.size):
            
            ax.plot(np.ones(2)*self.trans_ticks[i], ylim,
                    color='grey', zorder=0, ls='--')
            
            ax.plot(np.ones(2)*self.base_ticks[i], ylim,
                    color='cyan', zorder=0, ls='--')
            
            ax.text(self.base_ticks[i]+self.Tblank/10, ylim[-1]-D, "base")
            
            ax.plot(np.ones(2)*self.stim_ticks[i], ylim,
                    color='magenta', zorder=0, ls='--')  
            
            ax.text(self.base_ticks[i]+self.Tblank+self.Tstim/10, ylim[-1]-D,
                    "pert\n{:.0f}%".format(nn_stim_rng[i]/self.NI*100))
            
        ax.set_ylim(ylim)
        
    def get_indegree(self):
        
        self.indeg_etoe = (self.w_etoe>0).sum(axis=0)
        self.indeg_etoi = (self.w_etoi>0).sum(axis=0)
        self.indeg_itoe = (self.w_itoe>0).sum(axis=0)
        self.indeg_itoi = (self.w_itoi>0).sum(axis=0)
        
        self.sum_w_etoe = self.w_etoe.sum(axis=0)
        self.sum_w_etoi = self.w_etoi.sum(axis=0)
        self.sum_w_itoe = self.w_itoe.sum(axis=0)
        self.sum_w_itoi = self.w_itoi.sum(axis=0)
        
    def plot_reg_line(self, x, y, ax):
        
        x_line = np.array([x.min(), x.max()])
        
        out = linregress(x, y)
        y_line = out.slope*x_line + out.intercept
        
        ax.plot(x_line, y_line, color='black')
        #ax.set_title('r={:.2f}-'.format(out.pvalue)+ax.get_title())
        
    def plot_basefr_frdiff(self, ax):
        
        """
        Method for visualizing the relationship between changes of 
        baseline firing rate against firing rate changes of both populations.
        
        Parameters: 
            
            ax: numpy array, list
                axes in which scatter plots will appear. First element refers 
                to the top and second to the bottom panel in a figure with
                2-by-n subplots.
            
            EI: list of strings (default: "I")
                for which neural population the visualization should be. It can
                be either "I" or "E"
        
        """
        
        ax[0].scatter(self.diff_inh, self.base_inh_nz, s=1, color='blue', label='I')
        ax[1].scatter(self.diff_exc, self.base_exc_nz, s=1, color='red', label='E')
        self.plot_reg_line(self.diff_inh, self.base_inh_nz, ax[0])
        self.plot_reg_line(self.diff_exc, self.base_exc_nz, ax[1])
        
    def plot_gdiff_frdiff(self, ax, EI="I"):
        
        """
        Method for visualizing the relationship between changes of 
        inhibitory/excitatory conductances against firing rate changes of their
        target neurons.
        
        Parameters: 
            
            ax: numpy array, list
                axes in which scatter plots will appear. First element refers 
                to the top and second to the bottom panel in a figure with
                2-by-n subplots.
            
            EI: list of strings (default: "I")
                for which neural population the visualization should be. It can
                be either "I" or "E"
        
        """
        
        if EI == "I":
            dfr = self.diff_inh
            c = "blue"
            nonzero = (self.base_inh!=0) | (self.stim_inh!=0)
        elif EI == "E":
            dfr = self.diff_exc
            c = "red"
            nonzero = (self.base_exc!=0) | (self.stim_exc!=0)
        else:
            pass #rise error
        
        gidiff = self.diff_cond[EI]["gi"][nonzero]
        gediff = self.diff_cond[EI]["ge"][nonzero]
        
        ax[0].scatter(dfr, gidiff, s=1, color=c)
        ax[1].scatter(dfr, gediff, s=1, color=c)
        self.plot_reg_line(dfr, gidiff, ax[0])
        self.plot_reg_line(dfr, gediff, ax[1])
        
        
    def plot_indeg_frdiff(self, ax):
        
        ax.scatter(self.diff_inh_m, self.indeg_etoi, s=1)
        self.plot_reg_line(self.diff_inh_m, self.indeg_etoi, ax)
        
    def plot_indeg_frdiff_e(self, ax):
        
        ax.scatter(self.diff_exc_m, self.indeg_etoe, s=1)
        self.plot_reg_line(self.diff_exc_m, self.indeg_etoe, ax)
        
    def plot_inpfr_frdiff(self, ax):

        sum_fr_e = np.zeros_like(self.stim_inh)
        sum_fr_i = np.zeros_like(self.stim_inh)
        w_etoi_bin = self.w_etoi>0
        w_itoi_bin = self.w_itoi<0
        
        for i in range(self.NI):
            
            sum_fr_e[i, :] = self.stim_exc[w_etoi_bin[:, i], :].sum(axis=0)
            sum_fr_i[i, :] = self.stim_inh[w_itoi_bin[:, i], :].sum(axis=0)
            
        sum_fr_e = sum_fr_e[(self.base_inh!=0) | (self.stim_inh!=0)]
        sum_fr_i = sum_fr_i[(self.base_inh!=0) | (self.stim_inh!=0)]
        
        ax[0].scatter(self.diff_inh.flatten(), sum_fr_i.flatten(),
                      color='blue', s=1)
        ax[1].scatter(self.diff_inh.flatten(), sum_fr_e.flatten(),
                      color='red', s=1)
        self.plot_reg_line(self.diff_inh.flatten(), sum_fr_i.flatten(), ax[0])
        self.plot_reg_line(self.diff_inh.flatten(), sum_fr_e.flatten(), ax[1])
        
    def plot_inpfr_frdiff_e(self, ax):

        sum_fr_e = np.zeros_like(self.stim_exc)
        sum_fr_i = np.zeros_like(self.stim_exc)
        w_etoe_bin = self.w_etoe>0
        w_itoe_bin = self.w_itoe<0
        
        for i in range(self.NE):
            
            sum_fr_e[i, :] = self.stim_exc[w_etoe_bin[:, i], :].sum(axis=0)
            sum_fr_i[i, :] = self.stim_inh[w_itoe_bin[:, i], :].sum(axis=0)
            
        sum_fr_e = sum_fr_e[(self.base_exc!=0) | (self.stim_exc!=0)]
        sum_fr_i = sum_fr_i[(self.base_exc!=0) | (self.stim_exc!=0)]
        
        ax[0].scatter(self.diff_exc.flatten(), sum_fr_i.flatten(),
                      color='blue', s=1)
        ax[1].scatter(self.diff_exc.flatten(), sum_fr_e.flatten(),
                      color='red', s=1)
        self.plot_reg_line(self.diff_exc.flatten(), sum_fr_i.flatten(), ax[0])
        self.plot_reg_line(self.diff_exc.flatten(), sum_fr_e.flatten(), ax[1])
        
    def plot_frdiffmean_dist(self, ax, pert_num, num_bins=20):
        
        # edges = np.linspace(self.diff_inh_m.min(), self.diff_inh_m.max(), num_bins)
        dim_not_nan = self.diff_inh_m[np.logical_not(np.isnan(self.diff_inh_m))]
        c_inh, e_inh = np.histogram(self.diff_inh_m, num_bins,
                                    range=(dim_not_nan.min(), dim_not_nan.max()))
        e_inh = (e_inh[0:-1]+e_inh[1:])/2
        dem_not_nan = self.diff_exc_m[np.logical_not(np.isnan(self.diff_exc_m))]
        c_exc, e_exc = np.histogram(self.diff_exc_m, num_bins,
                                    range=(dem_not_nan.min(), dem_not_nan.max()))
        e_exc = (e_exc[0:-1]+e_exc[1:])/2
        
        if hasattr(self, 'diff_inh_pv'):
            print('NotImplemented!')
        else:
            ax.plot(e_inh, c_inh, color='blue', label=r'$I$')
            ax.plot(e_exc, c_exc, color='red', label=r'$E$')
            # ax.hist([self.diff_inh_m,
            #          self.diff_exc_m],
            #          num_bins,
            #          color=['blue', 'red'],
            #          label=[r'$I$', r'$E$'])
            
    def plot_frdiffmean_dist_pertdistinct_line_by_model(self, pert_val,
                                                        fig_ca, fig_dir,
                                                        num_bins=20):
        fig_dir = self.create_fig_subdir(fig_dir, "dist-line-bymodel-{:.2f}-{:.2}".format(self.Be, self.Bi))
        # fig_dir = self.create_fig_subdir(fig_dir, "{:.0f}".format(pert_num))
        # edges = np.linspace(self.diff_inh_m.min(), self.diff_inh_m.max(), num_bins)
        num_rows = 2
        fig, ax = plt.subplots(nrows = num_rows, ncols = self.num_models//num_rows,
                               sharey=True)
        
        for mdl_i in range(self.num_models):
            row = mdl_i%num_rows
            col = mdl_i//num_rows
            diff_inh_m = self.diff_inh_m[:, mdl_i]
            diff_exc_m = self.diff_exc_m[:, mdl_i]
        
            if fig_ca == 'ca3':
                inh_pert = diff_inh_m[0:pert_val]
                inh_nonpert = diff_inh_m[pert_val:]
                exc_pert = diff_exc_m[0:int(pert_val*NE/NI)]
                exc_nonpert = diff_exc_m[int(pert_val*NE/NI):]
            elif fig_ca == 'ca1':
                inh_pert, inh_nonpert = np.array([]), diff_inh_m
                exc_pert, exc_nonpert = np.array([]), diff_exc_m
        
            inh_exc = (inh_pert, inh_nonpert, exc_pert, exc_nonpert)
            num_neurons = (NI, NI, NE, NE)
            
            max_ = -np.inf
            min_ = np.inf
            for d in inh_exc:
                if d.size > 0:
                    max_ = max(max_, d.max())
                    min_ = min(min_, d.min())
        
            edges = np.linspace(min_, max_, num=num_bins)
            mid_points = (edges[1:] + edges[:-1])/2
            
            c_inh_exc = []
            
            for i, c in enumerate(inh_exc):
                if fig_ca=='ca1':
                    if c.size > 0:
                        c_inh_exc.append(np.histogram(c, edges)[0]/num_neurons[i])
                    else:
                        c_inh_exc.append(np.array([]))
                else:
                    c_inh_exc.append(np.histogram(c, edges)[0]/num_neurons[i])
            
            L = [r'$I_{pert}$', r'$I$', r'$E_{pert}$', r'$E$']
            Col = ['blue', 'skyblue', 'red', 'lightsalmon']
            
            if hasattr(self, 'diff_inh_pv'):
                print('NotImplemented!')
            else:
                for i, c in enumerate(c_inh_exc):
                    if c.size > 0:
                        ax[row, col].plot(mid_points, c,
                                          color=Col[i],
                                          label=L[i])
            if row == 2:
                ax[row, col].set_xlabel(r'$\Delta FR(spk/s)$')
            if col == 0:
                ax[row, col].set_ylabel('Proportion')
        ax[-1, -1].legend(bbox_to_anchor=(1., 1),
                      loc='upper left',
                      borderaxespad=0.)
        #to_square_plots(ax)
        boxoff(ax)
        fig.suptitle("Photoinhibiting {:.0f}% of CA3 neurons".format(pert_val/self.NI*100))
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "{:.0f}.pdf".format(pert_val)),
                                 format="pdf")
        plt.close(fig)

    def plot_frdiffmean_dist_pertdistinct_line(self, ax, pert_num, fig_ca, num_bins=20):
        
        # edges = np.linspace(self.diff_inh_m.min(), self.diff_inh_m.max(), num_bins)
        
        if fig_ca == 'ca3':
            inh_pert, inh_nonpert = self.diff_inh_m[0:pert_num].flatten(), self.diff_inh_m[pert_num:].flatten()
            exc_pert, exc_nonpert = self.diff_exc_m[0:int(pert_num*NE/NI)].flatten(), self.diff_exc_m[int(pert_num*NE/NI):].flatten()
        elif fig_ca == 'ca1':
            if pert_pop == 'ca1':
                inh_pert, inh_nonpert = self.diff_inh_m[0:pert_num].flatten(), self.diff_inh_m[pert_num:].flatten()
                exc_pert, exc_nonpert = self.diff_exc_m[0:int(pert_num*NE/NI)].flatten(), self.diff_exc_m[int(pert_num*NE/NI):].flatten()
            else:
                inh_pert, inh_nonpert = np.array([]), self.diff_inh_m.flatten()
                exc_pert, exc_nonpert = np.array([]), self.diff_exc_m.flatten()
        
        inh_exc = (inh_pert, inh_nonpert, exc_pert, exc_nonpert)
        num_neurons = (NI*self.num_models, NI*self.num_models, NE*self.num_models, NE*self.num_models)
        max_ = -np.inf
        min_ = np.inf
        for d in inh_exc:
            if d.size > 0:
                max_ = max(max_, d.max())
                min_ = min(min_, d.min())
    
        # max_ = max((inh_pert.max(), inh_nonpert.max(), exc_pert.max(), exc_nonpert.max()))
        # min_ = min((inh_pert.min(), inh_nonpert.min(), exc_pert.min(), exc_nonpert.min()))
        edges = np.linspace(min_, max_, num=num_bins)
        mid_points = (edges[1:] + edges[:-1])/2
        
        c_inh_exc = []
        
        for i, c in enumerate(inh_exc):
            if (fig_ca=='ca1') & (pert_pop!='ca1'):
                if c.size > 0:
                    c_inh_exc.append(np.histogram(c, edges)[0]/num_neurons[i])
                else:
                    c_inh_exc.append(np.array([]))
            else:
                c_inh_exc.append(np.histogram(c, edges)[0]/num_neurons[i])
        
        # c_inh_pert = np.histogram(inh_pert, edges)[0]
        # c_inh_nonpert = np.histogram(inh_nonpert, edges)[0]
        # c_exc_pert = np.histogram(exc_pert, edges)[0]
        # c_exc_nonpert = np.histogram(exc_nonpert, edges)[0]
        
        # C = [c_inh_pert, c_inh_nonpert, c_exc_pert, c_exc_nonpert]
        L = [r'$I_{pert}$', r'$I$', r'$E_{pert}$', r'$E$']
        Col = ['blue', 'skyblue', 'red', 'lightsalmon']
        
        if hasattr(self, 'diff_inh_pv'):
            print('NotImplemented!')
        else:
            for i, c in enumerate(c_inh_exc):
                if c.size > 0:
                    ax.plot(mid_points, c,
                            color=Col[i],
                            label=L[i])
            
    def plot_frdiffmean_dist_pertdistinct(self, ax, pert_num, fig_ca, num_bins=20):
        
        # edges = np.linspace(self.diff_inh_m.min(), self.diff_inh_m.max(), num_bins)
        
        if hasattr(self, 'diff_inh_pv'):
            print('NotImplemented!')
        else:
            if fig_ca == 'ca1':
                ax.hist([self.diff_inh_m.flatten(),
                         self.diff_exc_m.flatten()],
                         num_bins,
                         color=['skyblue', 'lightsalmon'],
                         label=[r'$I$', r'$E$'])
            elif fig_ca == 'ca3':
                ax.hist([self.diff_inh_m[0:pert_num].flatten(),
                         self.diff_inh_m[pert_num:].flatten(),
                         self.diff_exc_m[0:int(pert_num*NE/NI)].flatten(),
                         self.diff_exc_m[int(pert_num*NE/NI):].flatten()],
                         num_bins,
                         color=['blue', 'skyblue', 'red', 'lightsalmon'],
                         label=[r'$I_{pert}$', r'$I$', r'$E_{pert}$', r'$E$'])
            
    def plot_frdiffmean_samplesize_dist(self, ax, num_bins=20):
        
        # edges = np.linspace(self.diff_inh_m.min(), self.diff_inh_m.max(), num_bins)
        
        if hasattr(self, 'diff_inh_pv'):
            print('NotImplemented!')
        else:
            ax.hist([self.diff_inh_m_include_per.flatten(),
                     self.diff_exc_m_include_per.flatten()], self.Ntrials,
                    color=['blue', 'red'],
                    label=['I', 'E'])
        
    def plot_frdiff_dist(self, ax, num_bins=20):
        
        edges = np.linspace(self.diff_inh.min(), self.diff_inh.max(), num_bins)
        
        if hasattr(self, 'diff_inh_pv'):
            ax.hist([self.diff_inh_pv.flatten(),
                     self.diff_inh_som.flatten(),
                     self.diff_inh_vip.flatten(),
                     self.diff_exc.flatten()], edges,
                    color=[(1,1,.2), (1,1,.5), (0,0,.8), (1,0,0)],
                    label=['I_pv', 'I_som', 'I_vip', 'E'])
        else:
            ax.hist([self.diff_inh.flatten(),
                     self.diff_exc.flatten()], edges,
                    color=['blue', 'red'],
                    label=['I', 'E'])
        
    def plot_conddiff_dist(self, ax, num_bins=20):
        
        edges = np.linspace(self.diff_i_cond.min(), self.diff_i_cond.max(), num_bins)
        ax.hist([self.diff_i_cond.flatten(),
                 self.diff_e_cond.flatten()], edges,
                color=['blue', 'red'],
                label=['I', 'E'])
        
    def plot_box_frdiff(self, ax, pert_val):
        
        pert_percent = int(pert_val/self.NI*100)
        
        if self.diff_inh.max()<50:
        
            ax[0].boxplot(self.diff_inh.flatten(), positions=[pert_percent],
                          widths=[10], flierprops={'marker': '.'})
            
        if self.diff_exc.max()<50:
            
            ax[1].boxplot(self.diff_exc.flatten(), positions=[pert_percent],
                          widths=[10], flierprops={'marker': '.'})
        
    def plot_violin_frdiff(self, ax, pert_val):
        
        # pert_percent = int(pert_val/self.NI*100)
        pos = np.where(nn_stim_rng==pert_val)[0]
        
        if self.diff_inh.max()<50:
            
            ax[0].violinplot([self.diff_inh.flatten()], positions=pos,
                             showextrema=False, showmeans=True)
            
        if self.diff_exc.max()<50:
            
            ax[1].violinplot([self.diff_exc.flatten()], positions=pos,
                             showextrema=False, showmeans=True)
                                     
        if pert_val == nn_stim_rng[-1]:
            xlabels = np.arange(nn_stim_rng.size)
            pert_percent = nn_stim_rng/self.NI*100
            pert_percent = pert_percent.astype('int')
            pert_perc_str = ['{:d}%'.format(pert) for pert in pert_percent]
            
            for a in ax:
                a.set_xticks(xlabels)
                a.set_xticklabels(pert_perc_str)
                # a.set_axis_style(a, labels)
        
    def plot_box_frdiffmean(self, ax, pert_val):
        
        pert_percent = int(pert_val/self.NI*100)
        
        if self.diff_inh.max()<50:
        
            ax[0].boxplot(self.diff_inh_m[~np.isnan(self.diff_inh_m)], positions=[pert_percent],
                           widths=[10], flierprops={'marker': '.'})
                          
        if self.diff_exc.max()<50:
        
            ax[1].boxplot(self.diff_exc_m[~np.isnan(self.diff_exc_m)], positions=[pert_percent],
                           widths=[10], flierprops={'marker': '.'})
        
    def plot_violin_frdiffmean(self, ax, pert_val):
        
        # pert_percent = int(pert_val/self.NI*100)
        pos = np.where(nn_stim_rng==pert_val)[0]
        
        if self.diff_inh.max()<50:
            
            ax[0].violinplot([self.diff_inh_m[~np.isnan(self.diff_inh_m)]], positions=pos,
                         showextrema=False, showmeans=True)
            
        if self.diff_exc.max()<50:
        
            ax[1].violinplot([self.diff_exc_m[~np.isnan(self.diff_exc_m)]], positions=pos,
                         showextrema=False, showmeans=True)
            
        if pert_val == nn_stim_rng[-1]:
            xlabels = np.arange(nn_stim_rng.size)
            pert_percent = nn_stim_rng/self.NI*100
            pert_percent = pert_percent.astype('int')
            pert_perc_str = ['{:d}%'.format(pert) for pert in pert_percent]
            
            for a in ax:
                a.set_xticks(xlabels)
                a.set_xticklabels(pert_perc_str)
        
    def plot_box_conddiff(self, ax, pert_val):
        
        pert_percent = int(pert_val/self.NI*100)

        ax[0].boxplot(self.diff_i_cond.flatten(), positions=[pert_percent],
                       widths=[10], flierprops={'marker': '.'})
        
        ax[1].boxplot(self.diff_e_cond.flatten(), positions=[pert_percent],
                       widths=[10], flierprops={'marker': '.'})
        
    def plot_fr_dist(self, ax, num_bins=20):
        
        _max = max(self.base_exc.max(), self.base_inh.max())
        _min = min(self.base_exc.min(), self.base_inh.min())
        
        edges = np.linspace(_min, _max, num_bins)
        
        ax.hist([self.base_exc.flatten(),  
                 self.base_inh.flatten()], edges,
                color=['red', 'blue'],
                label=['E', 'I'])
        
    def create_fig_subdir(self, path, dir_name):
        
        dir_path = os.path.join(path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        
        return dir_path
    
    def plot_raster_tr(self, ids, times, ax):
        
        for i, e_id in enumerate(self.vis_E_ids):
            
            sel_spks = times[ids==e_id]
            ax.plot(sel_spks-self.Ttrans, (i+1)*np.ones_like(sel_spks),
                    color='red', marker='|', markersize=1, linestyle='')
            
        for i, i_id in enumerate(self.vis_I_ids):
            
            sel_spks = times[ids==i_id]
            ax.plot(sel_spks-self.Ttrans, (i+self.vis_E_ids.size+1)*np.ones_like(sel_spks),
                    color='blue', marker='|', markersize=1, linestyle='')
    # def plot_raster_tr_sep(self, ids, times, ax, sel_ids):
        
    #     if self.sel_ids.size != 0:
        
    #         for i, s_id in enumerate(sel_ids):
                
    #             sel_spks = times[ids==s_id]
                
    #             ax.plot(sel_spks, (i+1)*np.ones_like(sel_spks),
    #                     color='red', marker='|', markersize=1, linestyle='')
            
    def get_ids_of_changing_units(self, th_prop_trials=0.5):
        
        pos_exc_bin = self.diff_exc>0
        pos_inh_bin = self.diff_inh>0
        
        neg_exc_bin = self.diff_exc<0
        neg_inh_bin = self.diff_inh<0
        
        pos_exc = pos_exc_bin.sum(axis=1)
        pos_inh = pos_inh_bin.sum(axis=1)
        
        neg_exc = neg_exc_bin.sum(axis=1)
        neg_inh = neg_inh_bin.sum(axis=1)
        
        self.highest_pos_exc = pos_exc.argmax()+1
        self.highest_pos_inh = pos_inh.argmax()+self.NE+1
        
        self.highest_neg_exc = neg_exc.argmax()+1
        self.highest_neg_inh = neg_inh.argmax()+self.NE+1
        
        self.pos_exc_ids = np.where(pos_exc>th_prop_trials*self.Ntrials)[0]
        self.pos_inh_ids = np.where(pos_inh>th_prop_trials*self.Ntrials)[0]
        
        self.neg_exc_ids = np.where(neg_exc>th_prop_trials*self.Ntrials)[0]
        self.neg_inh_ids = np.where(neg_inh>th_prop_trials*self.Ntrials)[0]
        
        self.pos_exc_tr  = np.zeros_like(self.pos_exc_ids)
        self.pos_inh_tr  = np.zeros_like(self.pos_inh_ids)
        
        self.neg_exc_tr  = np.zeros_like(self.neg_exc_ids)
        self.neg_inh_tr  = np.zeros_like(self.neg_inh_ids)
        
        for i, a in enumerate(self.diff_exc[self.pos_exc_ids]):
            self.pos_exc_tr[i] = a.argmax()
            
        for i, a in enumerate(self.diff_inh[self.pos_inh_ids]):
            self.pos_inh_tr[i] = a.argmax()
            
        for i, a in enumerate(self.diff_exc[self.neg_exc_ids]):
            self.neg_exc_tr[i] = a.argmin()
            
        for i, a in enumerate(self.diff_inh[self.neg_inh_ids]):
            self.neg_inh_tr[i] = a.argmin()
            
    def plot_raster_sample_high_chgs(self, pert_val, ax, ax_rate):
        
        if not hasattr(self, 'pos_exc_ids'):
            self.get_ids_of_changing_units()
        
        self.plot_raster_sample_chgs(pert_val, ax[0, 0], ax_rate[0, 0],
                                     self.highest_pos_exc, color='black')
        
        self.plot_raster_sample_chgs(pert_val, ax[0, 1], ax_rate[0, 1],
                                     self.highest_pos_inh, color='black')
            
        self.plot_raster_sample_chgs(pert_val, ax[1, 0], ax_rate[1, 0],
                                     self.highest_neg_exc, color='black')
        
        self.plot_raster_sample_chgs(pert_val, ax[1, 1], ax_rate[1, 1],
                                     self.highest_neg_inh, color='black')
        
        ax[1, 0].set_xlabel("Time (ms)")
        ax[1, 1].set_xlabel("Time (ms)")
        ax[0, 0].set_ylabel("Trial ID")
        ax[1, 0].set_ylabel("Trial ID")
        ax[0, 0].set_title("Excitatory")
        ax[0, 1].set_title("Inhibitory")
        
        ax_rate[1, 0].set_xlabel("Time (ms)")
        ax_rate[1, 1].set_xlabel("Time (ms)")
        ax_rate[0, 0].set_title("Excitatory")
        ax_rate[0, 1].set_title("Inhibitory")
        ax_rate[0, 0].set_ylabel("Firing rate (spk/s)")
        ax_rate[1, 0].set_ylabel("Firing rate (spk/s)")
    def plot_raster_sample_chgs(self, pert_val, ax, ax_rate, sel_ids, color):
        
        spk_times = np.array([])
        T_vec = np.arange(self.Ttrans, self.Texp+1)
        T = (T_vec[1:]+T_vec[0:-1])/2
        
        for tr in range(self.Ntrials):
            
            if self.trial_type == 'SingleSim':
            
                spk_t = spk_times[(spk_times>=self.st_tr_time[tr]) & 
                                  (spk_times<=self.end_tr_time[tr])] - self.st_tr_time[tr]
                
                # spk_id = spk_ids[(spk_times>=self.st_tr_time[tr]) & 
                #                   (spk_times<=self.end_tr_time[tr])]
                
            else:
                
                spk_t = self.sim_res[pert_val][2][tr]['times']
                spk_id = self.sim_res[pert_val][2][tr]['senders']
                sel_spks = spk_t[spk_id==sel_ids]
                spk_times = np.concatenate((spk_times, sel_spks[sel_spks>self.Tstim]))
                ax.plot(sel_spks-self.Ttrans, (tr+1)*np.ones_like(sel_spks),
                        color=color, marker='|',
                        markersize=1, linestyle='', zorder=1)
                        
        ax.axvspan(self.Tblank, self.Tblank+self.Tstim,\
				   zorder=0, color=[0.7, 0.7, 0.7])
        ax.set_xlim((0, self.Tblank*2+self.Tstim))	
        cnts = np.histogram(np.array(spk_times), T_vec)[0]/self.Ntrials*1000
        ax_rate.plot(T-self.Ttrans, self.smooth(cnts, win_size=100), zorder=1)   
        #ax_rate.set_xlim((0, self.Tblank*2+self.Tstim))
        ax_rate.axvspan(self.Tblank, self.Tblank+self.Tstim,\
                        zorder=0, color=[0.7, 0.7, 0.7])
        ax_rate.set_xlim((0, self.Tstim+2*self.Tblank))
        boxoff([ax_rate, ax])
    def plot_raster_abs_chgs_all(self, pert_val, ax):
        
        if not hasattr(self, 'pos_exc_ids'):
            self.get_ids_of_changing_units()
        
        self.plot_raster_abs_chgs(pert_val, ax[0, 0],
                                  self.pos_exc_ids+1, self.pos_exc_tr, color='red')
        
        self.plot_raster_abs_chgs(pert_val, ax[0, 1],
                                  self.pos_inh_ids+1+self.NE, self.pos_inh_tr, color='blue')
        
        self.plot_raster_abs_chgs(pert_val, ax[1, 0],
                                  self.neg_exc_ids+1, self.neg_exc_tr, color='red')
        
        self.plot_raster_abs_chgs(pert_val, ax[1, 1],
                                  self.neg_inh_ids+1+self.NE, self.neg_inh_tr, color='blue')
        
        
        ax[1, 0].set_xlabel("Time (ms)")
        ax[1, 1].set_xlabel("Time (ms)")
        ax[0, 0].set_ylabel("Neuron ID")
        ax[1, 0].set_ylabel("Neuron ID")
        ax[0, 0].set_title("Excitatory")
        ax[0, 1].set_title("Inhibitory")
            
    def plot_raster_abs_chgs(self, pert_val, ax, sel_ids, sel_tr, color):
        
        for i, tr in enumerate(sel_tr):
            
            if self.trial_type == 'SingleSim':
            
                spk_t = spk_times[(spk_times>=self.st_tr_time[tr]) & 
                                  (spk_times<=self.end_tr_time[tr])] - self.st_tr_time[tr]
                
                spk_id = spk_ids[(spk_times>=self.st_tr_time[tr]) & 
                                 (spk_times<=self.end_tr_time[tr])]
                
            else:
                
                spk_t = self.sim_res[pert_val][2][tr]['times']
                spk_id = self.sim_res[pert_val][2][tr]['senders']
                
                sel_spks = spk_t[spk_id==sel_ids[i]]
                
                ax.plot(sel_spks, (i+1)*np.ones_like(sel_spks),
                        color=color, marker='|', markersize=1, linestyle='')
        
    def plot_raster(self, pert_val, ax, prop=0.05):
        
        if not hasattr(self, 'st_tr_time'):
            self.get_trial_times()
            
        vis_NE = int(self.NE*prop)
        vis_NI = int(self.NI*prop)
            
        if self.trial_type == 'SingleSim':
            
            spk_times = self.sim_res[pert_val][2]['times']
            spk_ids   = self.sim_res[pert_val][2]['senders']
        
            self.vis_E_ids = np.random.choice(np.unique(spk_ids[spk_ids<=self.NE]),
                                              vis_NE, replace=False)
            self.vis_I_ids = np.random.choice(np.unique(spk_ids[spk_ids>self.NE]),
                                              vis_NI, replace=False)
            
        else:
            
            self.vis_E_ids = np.random.choice(np.arange(self.NE), vis_NE, replace=False)
            self.vis_I_ids = np.random.choice(np.arange(self.NE, self.Nall), vis_NI, replace=False)
            
        for i in range(self.Ntrials):
            
            if self.trial_type == 'SingleSim':
            
                spk_t = spk_times[(spk_times>=self.st_tr_time[i]) & 
                                  (spk_times<=self.end_tr_time[i])] - self.st_tr_time[i]
                
                spk_id = spk_ids[(spk_times>=self.st_tr_time[i]) & 
                                 (spk_times<=self.end_tr_time[i])]
                
            else:
                
                spk_t = self.sim_res[pert_val][2][i]['times']
                spk_id = self.sim_res[pert_val][2][i]['senders']
            
            self.plot_raster_tr(spk_id, spk_t, ax[i])
            
        ax[-1].set_xlabel("Time (ms)")
        ax[2].set_ylabel("Neuron ID")
        
''' Functions '''
def frchg_vs_EtoI(data, ref_cond="E"):
    
    '''
    Function for analyzing the firing rate changes as a function of balance
    between excitation and inhibition.
    
    Parameters:
    -----------
    
    data : dict (nested, values 3D numpy array)
        contains "I" and "E" keys that points to inhibitory and excitatory
        populations respectively. Under each key there are two dictionaries
        with two keys: "mean_change" & "proportion_increase". "mean_change" has
        mean firing rate changes and "proportion_increase" has proportion of
        neurons increase their firing rates.
        
        Each key has 3D numpy array as value. 1st D corresponds to Be_rng,
        2nd to Bi_rng and 3rd to nn_stim_rng.
        
    ref_cond: str (default="E")
        specifying the color coding whether excitatory or inhibitory conductances
        
    Returns:
    --------
    fig : matplotlib object
        figure handle that can e.g. be used for saving the figure.        
    '''
    
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True,  sharey=True, figsize=(9,5))
    cax = fig.add_axes([.91, 0.2, 0.02, 0.5])
    
    if ref_cond == "E":
        for i in range(Bi_rng.size):
            for j in range(nn_stim_rng.size):
                EtoI = Be_rng/Bi_rng[i]*-1
                sc1 = ax[0, j].scatter(EtoI,
                                       data['I']['mean_change'][:, i, j],
                                       c=Be_rng,
                                       s=10)
                sc2 = ax[1, j].scatter(EtoI,
                                       data['E']['mean_change'][:, i, j],
                                       c=Be_rng,
                                       s=10)
    elif ref_cond == "I":
        for i in range(Be_rng.size):
            for j in range(nn_stim_rng.size):
                EtoI = Be_rng[i]/Bi_rng*-1
                sc1 = ax[0, j].scatter(EtoI,
                                       data['I']['mean_change'][i, :, j],
                                       c=-Bi_rng,
                                       s=10)
                sc2 = ax[1, j].scatter(EtoI,
                                       data['E']['mean_change'][i, :, j],
                                       c=-Bi_rng,
                                       s=10)
        
        
            
    fig.colorbar(sc1, cax=cax, orientation='vertical')
    ax[-1, 2].set_xlabel(r'$E/I$')
    ax[0, 0].set_ylabel(r'Mean $\Delta FR_I$ (spk/s)')
    ax[1, 0].set_ylabel(r'Mean $\Delta FR_E$ (spk/s)')
    # ax[1, 0].set_xticks(np.array([0., 1., 2.5]))
    for j, nn in enumerate(nn_stim_rng):
        ax[0, j].set_title('pert={:.0f}%'.format(nn/nn_stim_rng.max()*100))
    
    if ref_cond == "E":
        cax.set_ylabel('E conductance (nS)')
    elif ref_cond == "I":
        cax.set_ylabel('I conductance (nS)')
    return fig

def significant_proportions(data):
    
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True,  sharey=True, figsize=(7,8))

    # cax = fig.add_axes([.25, 0.92, 0.5, 0.02])
    for i in range(Bi_rng.size):
        for j in range(nn_stim_rng.size):
            ax[i, j].plot(Be_rng, data[0][:, i, j], color='red', label='sig. decreasing')
            ax[i, j].plot(Be_rng, data[1][:, i, j], color='blue', label='sig. increasing')
            ax[i, j].plot(Be_rng, data[2][:, i, j], color='black', label='not sig. changing')
            
    ax[-1, 2].set_xlabel(r'$E conductance (nS)$')
    ax[2, 0].set_ylabel('Proportion')
    # ax[1, 0].set_xticks(np.array([0., 1., 2.5]))
    for j, nn in enumerate(nn_stim_rng):
        ax[0, j].set_title('pert={:.0f}%'.format(nn/nn_stim_rng.max()*100))
    ax[0, -1].legend()       
    return fig

def propposfrchg(data):
    
    '''
    Function for visualizing the proportion of firing rate changes for excitatory
    and inihibitory populations
    
    Parameters:
    -----------
    
    data : dict (nested, values 3D numpy array)
        contains "I" and "E" keys that points to inhibitory and excitatory
        populations respectively. Under each key there are two dictionaries
        with two keys: "mean_change" & "proportion_increase". "mean_change" has
        mean firing rate changes and "proportion_increase" has proportion of
        neurons increase their firing rates.
        
        Each key has 3D numpy array as value. 1st D corresponds to Be_rng,
        2nd to Bi_rng and 3rd to nn_stim_rng.
        
    Returns:
    --------
    fig : matplotlib object
        figure handle that can e.g. be used for saving the figure.        
    '''
    
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True,  sharey=True, figsize=(7,8))
    cax = fig.add_axes([.25, 0.92, 0.5, 0.02])
    for i in range(Bi_rng.size):
        for j in range(nn_stim_rng.size):
            sc = ax[i, j].scatter(data['I']['proportion_increase'][:, i, j],
                                 data['E']['proportion_increase'][:, i, j],
                                 c=Be_rng,
                                 s=10)
            ax[i, j].plot([0, 100], [0, 100], color='red', linestyle='-.', linewidth=0.5)
            ax[i, j].plot([50, 50], [0, 100], color='blue', linestyle='-.', linewidth=0.5)
            ax[i, j].spines["right"].set_visible(False)
            ax[i, j].spines["top"].set_visible(False)
    fig.colorbar(sc, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    ax[-1, 2].set_xlim([0, 101])
    ax[-1, 2].set_ylim([0, 101])
    ax[-1, 2].set_xlabel('Proportion of positive changes in I (%)')
    ax[2, 0].set_ylabel('Proportion of positive changes in E (%)')
    for j, nn in enumerate(nn_stim_rng):
        ax[0, j].set_title('pert={:.0f}%'.format(nn/nn_stim_rng.max()*100))
    for i, bi in enumerate(Bi_rng):
        ax[i, -1].set_ylabel('Gi={:.1f}'.format(bi))
        ax[i, -1].yaxis.set_label_position('right')
    cax.set_xlabel('E conductance')
    return fig

def frchg(data):
    
    '''
    Function for visualizing the mean firing rate changes for excitatory
    and inihibitory populations
    
    Parameters:
    -----------
    
    data : dict (nested, values 3D numpy array)
        contains "I" and "E" keys that points to inhibitory and excitatory
        populations respectively. Under each key there are two dictionaries
        with two keys: "mean_change" & "proportion_increase". "mean_change" has
        mean firing rate changes and "proportion_increase" has proportion of
        neurons increase their firing rates.
        
        Each key has 3D numpy array as value. 1st D corresponds to Be_rng,
        2nd to Bi_rng and 3rd to nn_stim_rng.
        
    Returns:
    --------
    fig : matplotlib object
        figure handle that can e.g. be used for saving the figure.        
    '''
    
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True,  sharey=True, figsize=(7,8))
    cax = fig.add_axes([.25, 0.92, 0.5, 0.02])
    for i in range(Bi_rng.size):
        for j in range(nn_stim_rng.size):
            sc = ax[i, j].scatter(data['I']['mean_change'][:, i, j],
                                 data['E']['mean_change'][:, i, j],
                                 c=Be_rng,
                                 s=10)
            # ax[i, j].plot([0, 100], [0, 100], color='red', linestyle='-.', linewidth=0.5)
            # ax[i, j].plot([50, 50], [0, 100], color='blue', linestyle='-.', linewidth=0.5)
            ax[i, j].spines["right"].set_visible(False)
            ax[i, j].spines["top"].set_visible(False)
    fig.colorbar(sc, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    # ax[-1, 2].set_xlim([0, 101])
    # ax[-1, 2].set_ylim([0, 101])
    ax[-1, 2].set_xlabel(r'Mean $\Delta FR_I$ (spk/s)')
    ax[2, 0].set_ylabel(r'Mean $\Delta FR_E$ (spk/s)')
    for j, nn in enumerate(nn_stim_rng):
        ax[0, j].set_title('pert={:.0f}%'.format(nn/nn_stim_rng.max()*100))
    for i, bi in enumerate(Bi_rng):
        ax[i, -1].set_ylabel('Gi={:.1f}'.format(bi))
        ax[i, -1].yaxis.set_label_position('right')
    cax.set_xlabel('E conductance')
    return fig
    # fig.savefig('frchg-EtoI-E.pdf', format='pdf')

if __name__=='__main__':    
    
    cwd = os.getcwd()
    fig_path = os.path.join(cwd, fig_dir+sim_suffix)
    os.makedirs(fig_path, exist_ok=True)
    
    cv_e = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 3))
    cv_i = np.zeros_like(cv_e)
    
    ff_e = np.zeros_like(cv_e)
    ff_i = np.zeros_like(cv_e)
    
    pos_prop = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 2))
    mean_fr  = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 2))
    
    paradox_score = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size))
    
            
    for ij1, Be in enumerate(Be_rng):
        
        fig_box, ax_box = plt.subplots(nrows=2, ncols=Bi_rng.size,
                                       sharex=True, sharey=True,
                                       figsize=(6, 6))
        
        fig_box_g, ax_box_g = plt.subplots(nrows=2, ncols=Bi_rng.size,
                                       sharex=True, sharey=True)
        
        for ij2, Bi in enumerate(Bi_rng):
            
            os.chdir(os.path.join(cwd, res_dir + sim_suffix))
            sim_name = 'sim_res_Be{:.2f}_Bi{:.2f}'.format(Be, Bi)
            print('Reading {} ...\n'.format(sim_name))
            # fl = open(sim_name, 'rb'); sim_res = pickle.load(fl); fl.close()
            
            fig, ax = plt.subplots(nrows=2, ncols=3,
                                   sharex=True, sharey=True)
            fig_e, ax_e = plt.subplots(nrows=2, ncols=3,
                                       sharex=True, sharey=True)
            fig_base, ax_base = plt.subplots(nrows=2, ncols=3,
                                             sharex=True, sharey=True)
            fig_dist, ax_dist = plt.subplots(nrows=2, ncols=3,
                                             sharex=True, sharey=True)
            fig_dist_mean, ax_dist_mean = plt.subplots(nrows=2, ncols=3,
                                             sharex=True, sharey=True)
            fig_dist_g, ax_dist_g = plt.subplots(nrows=2, ncols=3,
                                                 sharex=True, sharey=True)
            fig_i_fr, ax_i_fr = plt.subplots(nrows=2, ncols=5,
                                             sharex=True, sharey='row')
            fig_e_fr, ax_e_fr = plt.subplots(nrows=2, ncols=5,
                                             sharex=True, sharey='row')
            fig_base_frdiff, ax_base_frdiff = plt.subplots(nrows=2, ncols=5,
                                                           sharex=True, sharey=True)
            fig_dg_dfrI, ax_dg_dfrI = plt.subplots(nrows=2, ncols=5,
                                                   sharex=True, sharey=True)
            fig_dg_dfrE, ax_dg_dfrE = plt.subplots(nrows=2, ncols=5,
                                                   sharex=True, sharey=True)
            
            fig_avg_fr, ax_avg_fr = plt.subplots()
            
            simdata_obj = simdata(sim_name)
            
            ax[0, 0].set_ylabel('E to I')
            ax[1, 0].set_ylabel('E to I')
            
            ax_i_fr[0, 0].set_ylabel(r'$\sum (spikes_I)$ to I')
            ax_i_fr[1, 0].set_ylabel(r'$\sum (spikes_E)$ to I')
            
            ax_e_fr[0, 0].set_ylabel(r'$\sum (spikes_I)$ to E')
            ax_e_fr[1, 0].set_ylabel(r'$\sum (spikes_E)$ to E')
            
            ax_e[0, 0].set_ylabel('E to E')
            ax_e[1, 0].set_ylabel('E to E')
            
            ax[1, 1].set_xlabel(r'$\Delta FR (sp/s)$')
            ax_e[1, 1].set_xlabel(r'$\Delta FR (sp/s)$')
            ax_i_fr[1, 2].set_xlabel(r'$\Delta FR (sp/s)$')
            ax_e_fr[1, 2].set_xlabel(r'$\Delta FR (sp/s)$')
            
            ax_base[1, 1].set_xlabel('Firing rate (sp/s)')
            
            ax_dist[1, 1].set_xlabel(r'$\Delta FR (sp/s)$')
            
            for ii, nn_stim in enumerate(nn_stim_rng):
                
                print('Analyzing simulation data for Be={}, Bi={} and pert={}'.format(Be, Bi, nn_stim))
                
                fig_raster, ax_raster = plt.subplots(nrows=Ntrials, ncols=1,
                                                     sharex=True, sharey=True)
                
                fig_raster_sep, ax_raster_sep = plt.subplots(nrows=2, ncols=2,
                                                             sharex=True, sharey=True)
                
                a_r, a_c = ii//3, ii%3
                
                simdata_obj.get_fr_diff(nn_stim)
                
                #if len(simdata_obj.sim_res[nn_stim])>3:
                if rec_from_cond:
                    simdata_obj.get_cond_diff(nn_stim)
                # simdata_obj.get_indegree()
                # simdata_obj.plot_indeg_frdiff(ax[a_r, a_c])
                # simdata_obj.plot_indeg_frdiff_e(ax_e[a_r, a_c])
                
                simdata_obj.plot_frdiff_dist(ax_dist[a_r, a_c])
                ax_dist[a_r, a_c].legend()
                
                simdata_obj.plot_frdiffmean_dist(ax_dist_mean[a_r, a_c])
                ax_dist_mean[a_r, a_c].legend()
                
                #simdata_obj.plot_conddiff_dist(ax_dist_g[a_r, a_c])
                
                paradox_score[ij1, ij2, ii] = simdata_obj.paradox_score
                
                simdata_obj.plot_fr_dist(ax_base[a_r, a_c])
                
                simdata_obj.plot_box_frdiff(ax_box[:, ij2], nn_stim)
                
                #simdata_obj.plot_box_conddiff(ax_box_g[:, ij2], nn_stim)
                
                simdata_obj.plot_basefr_frdiff(ax_base_frdiff[:, ii])
                ax_base_frdiff[0, ii].set_title("pert={:.0f}%".format(nn_stim/NI*100))
                
                if rec_from_cond:
                    simdata_obj.plot_gdiff_frdiff(ax_dg_dfrI[:, ii], EI="I")
                    ax_dg_dfrI[0, ii].set_title("pert={:.0f}%".format(nn_stim/NI*100))
                
                    simdata_obj.plot_gdiff_frdiff(ax_dg_dfrE[:, ii], EI="E")
                    ax_dg_dfrE[0, ii].set_title("pert={:.0f}%".format(nn_stim/NI*100))
                
                simdata_obj.plot_inpfr_frdiff(ax_i_fr[:, ii])
                ax_i_fr[0, ii].set_title("pert={:.0f}%".format(nn_stim/NI*100))
                
                simdata_obj.plot_inpfr_frdiff_e(ax_e_fr[:, ii])
                ax_e_fr[0, ii].set_title("pert={:.0f}%".format(nn_stim/NI*100))
                
                simdata_obj.get_avg_frs(nn_stim)
                simdata_obj.concat_avg_frs_perts(ii)
                
                pos_prop[ij1, ij2, ii, 0] = np.sum(simdata_obj.diff_exc>0)/\
                                            simdata_obj.diff_exc.size*100
                                            
                pos_prop[ij1, ij2, ii, 1] = np.sum(simdata_obj.diff_inh>0)/\
                                            simdata_obj.diff_inh.size*100                            
                
                mean_fr[ij1, ij2, ii, 0]  = simdata_obj.diff_exc.mean()
                mean_fr[ij1, ij2, ii, 1]  = simdata_obj.diff_inh.mean()
                
                # cv_e[ij1, ij2, ii, 0] = simdata_obj.trans_cv[0, :].mean()
                # cv_e[ij1, ij2, ii, 1] = simdata_obj.base_cv[0, :].mean()
                # cv_e[ij1, ij2, ii, 2] = simdata_obj.stim_cv[0, :].mean()
                
                # ff_e[ij1, ij2, ii, 0] = simdata_obj.trans_ff[0, :].mean()
                # ff_e[ij1, ij2, ii, 1] = simdata_obj.base_ff[0, :].mean()
                # ff_e[ij1, ij2, ii, 2] = simdata_obj.stim_ff[0, :].mean()
                
                # cv_i[ij1, ij2, ii, 0] = simdata_obj.trans_cv[1, :].mean()
                # cv_i[ij1, ij2, ii, 1] = simdata_obj.base_cv[1, :].mean()
                # cv_i[ij1, ij2, ii, 2] = simdata_obj.stim_cv[1, :].mean()
                
                # ff_i[ij1, ij2, ii, 0] = simdata_obj.trans_ff[1, :].mean()
                # ff_i[ij1, ij2, ii, 1] = simdata_obj.base_ff[1, :].mean()
                # ff_i[ij1, ij2, ii, 2] = simdata_obj.stim_ff[1, :].mean()
                '''
                path_raster_fig = simdata_obj.create_fig_subdir(fig_path, "raster_dir")
                simdata_obj.plot_raster(nn_stim, ax_raster)
                fig_raster.savefig(os.path.join(path_raster_fig,
                                                "Be{}-Bi{}-P{}.png".format(Be, Bi, nn_stim)),
                                    format="png")
                '''
                plt.close(fig_raster)
                
                
                path_raster_fig = simdata_obj.create_fig_subdir(fig_path, "raster_dir_sep")
                # simdata_obj.plot_raster_abs_chgs_all(nn_stim, ax_raster_sep)
                simdata_obj.plot_raster_sample_high_chgs(nn_stim, ax_raster_sep)
                fig_raster.savefig(os.path.join(path_raster_fig,
                                                "Be{}-Bi{}-P{}.png".format(Be, Bi, nn_stim)),
                                    format="png")
                plt.close(fig_raster_sep)
                
                ax[a_r, a_c].set_title('P={}'.format(nn_stim))
                ax_dist[a_r, a_c].set_title('P={:.0f}%'.format(nn_stim/NI*100))
                ax_dist_mean[a_r, a_c].set_title('P={:.0f}%'.format(nn_stim/NI*100))
                ax_base[a_r, a_c].set_title('P={:.0f}%'.format(nn_stim/NI*100))
                
            
            ax_base_frdiff[1, 2].set_xlabel("Baseline firing rate (sp/s)")
            ax_base_frdiff[1, 0].set_ylabel(r"$\Delta FR_E (sp/s)$")
            ax_base_frdiff[0, 0].set_ylabel(r"$\Delta FR_I (sp/s)$")
            
            ax_dg_dfrI[1, 2].set_xlabel(r"$\Delta FR_I (sp/s)$")
            ax_dg_dfrI[1, 0].set_ylabel(r"$\Delta g_E$")
            ax_dg_dfrI[0, 0].set_ylabel(r"$\Delta g_I$")
            
            ax_dg_dfrE[1, 2].set_xlabel(r"$\Delta FR_E (sp/s)$")
            ax_dg_dfrE[1, 0].set_ylabel(r"$\Delta g_E$")
            ax_dg_dfrE[0, 0].set_ylabel(r"$\Delta g_I$")
            
            ax_box[0, ij2].set_title('Bi={}'.format(Bi))
            ax_box[1, ij2].xaxis.set_tick_params(rotation=90)
            
            simdata_obj.plot_avg_frs(ax_avg_fr)
            ax_avg_fr.set_title("Be={}, Bi={}".format(Be, Bi))
            ax_avg_fr.set_xlabel("Time (ms)")
            ax_avg_fr.set_ylabel("Average firing rate (sp/s)")
            
            fig.savefig(os.path.join(fig_path, "fr-Ninp-Be{}-Bi{}.pdf".format(Be, Bi)),
                        format="pdf")
            
            fig_e.savefig(os.path.join(fig_path, "fre-Ninp-Be{}-Bi{}.pdf".format(Be, Bi)),
                        format="pdf")
            
            fig_e_fr.suptitle("Excitatory neurons")
            fig_e_fr.savefig(os.path.join(fig_path, "fre-fr-diff-dist-Be{}-Bi{}.pdf".format(Be, Bi)),
                             format="pdf")
            
            fig_i_fr.suptitle("Inhibitory neurons")
            fig_i_fr.savefig(os.path.join(fig_path, "fr-fr-diff-dist-Be{}-Bi{}.pdf".format(Be, Bi)),
                             format="pdf")
            
            fig_dist.savefig(os.path.join(fig_path, "fr-diff-dist-Be{}-Bi{}.pdf".format(Be, Bi)),
                             format="pdf")
            
            fig_dist_mean.savefig(os.path.join(fig_path, "fr-diff-dist-mean-Be{}-Bi{}.pdf".format(Be, Bi)),
                                  format="pdf")
            
            fig_dist_g.savefig(os.path.join(fig_path, "g-diff-dist-Be{}-Bi{}.pdf".format(Be, Bi)),
                               format="pdf")
            
            fig_base.savefig(os.path.join(fig_path, "fr-base-dist-Be{}-Bi{}.pdf".format(Be, Bi)),
                             format="pdf")
            
            fig_avg_fr.savefig(os.path.join(fig_path, "avgfr-Be{}-Bi{}.pdf".format(Be, Bi)),
                               format="pdf")
            
            fig_base_frdiff.savefig(os.path.join(fig_path, "basevsdiff-Be{}-Bi{}.pdf".format(Be, Bi)),
                               format="pdf")
            
            fig_dg_dfrI.savefig(os.path.join(fig_path, "dfrvsd-I-Be{}-Bi{}.pdf".format(Be, Bi)),
                               format="pdf")
            
            fig_dg_dfrE.savefig(os.path.join(fig_path, "dfrvsd-E-Be{}-Bi{}.pdf".format(Be, Bi)),
                               format="pdf")
                    
            plt.close(fig)
            plt.close(fig_e)
            plt.close(fig_dist)
            plt.close(fig_dist_g)
            plt.close(fig_base)
            plt.close(fig_i_fr)
            plt.close(fig_e_fr)
            plt.close(fig_avg_fr)
            plt.close(fig_base_frdiff)
            plt.close(fig_dg_dfrI)
            plt.close(fig_dg_dfrE)
            
        ax_box[-1, 2].set_xlabel("Proportion of perturbed neurons (%)")
        ax_box[0, 0].set_ylabel(r"$\Delta FR_I$")
        ax_box[1, 0].set_ylabel(r"$\Delta FR_E$")
        
        ax_box_g[-1, 2].set_xlabel("Proportion of perturbed neurons (%)")
        ax_box_g[0, 0].set_ylabel(r"$\Delta g_I$")
        ax_box_g[1, 0].set_ylabel(r"$\Delta g_E$")
        
        fig_box.suptitle("Be={:.2f}".format(Be))
            
        fig_box.savefig(os.path.join(fig_path, "fr-diff-box-Be{}.pdf".format(Be)),
                         format="pdf")
        
        fig_box_g.savefig(os.path.join(fig_path, "g-diff-box-Be{}.pdf".format(Be)),
                          format="pdf")
        
        plt.close(fig_box)
        plt.close(fig_box_g)
        
        
    frchgdata = {'E': {'proportion_increase': pos_prop[:,:,:,0],
                       'mean_change': mean_fr[:,:,:,0]},
                 'I': {'proportion_increase': pos_prop[:,:,:,1],
                       'mean_change': mean_fr[:,:,:,1]}}
    
    fig_frchg_ei_e = frchg_vs_EtoI(frchgdata)
    fig_frchg_ei_e.savefig(os.path.join(fig_path, "frchg-EtoI-E.pdf"),
                           format="pdf")
    
    fig_frchg_ei_e = frchg_vs_EtoI(frchgdata, "I")
    fig_frchg_ei_e.savefig(os.path.join(fig_path, "frchg-EtoI-I.pdf"),
                           format="pdf")
    
    fig_posprop = propposfrchg(frchgdata)
    fig_posprop.savefig(os.path.join(fig_path, "propposfrchg.pdf"),
                        format="pdf")
    
    fig_frchg = frchg(frchgdata)
    fig_frchg.savefig(os.path.join(fig_path, "frchg.pdf"),
                        format="pdf")
    
    fl = open('fr-chgs-pos-prop', 'wb'); pickle.dump(frchgdata, fl); fl.close()
        
    '''    
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=False)
    for ij2 in range(Bi_rng.size):
        for ii in range(nn_stim_rng.size):
            
            ax[ij2, ii].scatter(pos_prop[:, ])
            simdata_obj.plot_pos_prop(, )
    '''        
        
    cv_ff_fig_path = os.path.join(fig_path, "CV-FF")
    os.makedirs(cv_ff_fig_path, exist_ok=True)
    
    Ge = np.append(0, Be_rng)
    Gi = np.append(0, Bi_rng)
    
    fig_psc, ax_psc = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    bar_psc = fig_psc.add_axes([0.7, 0.15, 0.02, 0.3])
        
    for ii, nn_stim in enumerate(nn_stim_rng):
        
        # if paradox_score[:, :, ii].max()>1:
        f = ax_psc[ii//3, ii%3].pcolormesh(Gi, Ge, paradox_score[:, :, ii],
                                           vmin=paradox_score.min(),
                                           vmax=paradox_score.max())
        ax_psc[ii//3, ii%3].set_title("pert={:.0f}%".format(nn_stim/NI*100))
        
    ax_psc[1, 0].set_xlabel("Inh. Cond.")
    ax_psc[1, 1].set_xlabel("Inh. Cond.")
    
    ax_psc[0, 0].set_ylabel("Exc. Cond.")
    ax_psc[1, 0].set_ylabel("Exc. Cond.")
    
    if "f" in locals():
        plt.colorbar(f, cax=bar_psc)
    
    fig_psc.savefig(os.path.join(fig_path, "paradoxical-score.pdf"), format="pdf")
        
        # fig_cv, ax_cv = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        # fig_ff, ax_ff = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        
        # for i in range(3):
        
        #     ce = ax_cv[0, i].pcolor(Bi_rng, Be_rng, cv_e[:, :, ii, i], vmin=0, vmax=cv_e.max())
        #     ci = ax_cv[1, i].pcolor(Bi_rng, Be_rng, cv_i[:, :, ii, i], vmin=0, vmax=cv_i.max())
            
        #     fe = ax_ff[0, i].pcolor(Bi_rng, Be_rng, ff_e[:, :, ii, i], vmin=ff_e.min(), vmax=ff_e.max())
        #     fi = ax_ff[1, i].pcolor(Bi_rng, Be_rng, ff_i[:, :, ii, i], vmin=ff_i.min(), vmax=ff_i.max())
        
        
        # fig_cv.subplots_adjust(right=0.8)
        # fig_ff.subplots_adjust(right=0.8)
        
        # cbar_cve = fig_cv.add_axes([0.85, 0.15, 0.02, 0.3])
        # cbar_cvi = fig_cv.add_axes([0.85, 0.5, 0.02, 0.3])
        # cbar_ffe = fig_ff.add_axes([0.85, 0.15, 0.02, 0.3])
        # cbar_ffi = fig_ff.add_axes([0.85, 0.5, 0.02, 0.3])
        
        # plt.colorbar(ce, cax=cbar_cve)
        # plt.colorbar(ci, cax=cbar_cvi)
        # plt.colorbar(fe, cax=cbar_ffe)
        # plt.colorbar(fi, cax=cbar_ffi)
        
        # ax_cv[1, 1].set_xlabel("Bi")
        # ax_cv[1, 0].set_ylabel("Be")
        # ax_cv[0, 0].set_ylabel("Be")
        
        # ax_ff[1, 1].set_xlabel("Bi")
        # ax_ff[1, 0].set_ylabel("Be")
        # ax_ff[0, 0].set_ylabel("Be")
        
        # fig_cv.suptitle("Coefficient of variation")
        # fig_ff.suptitle("Fano factor")
        
        # fig_cv.savefig(os.path.join(cv_ff_fig_path, "CV-P{}.pdf".format(nn_stim)))
        # fig_ff.savefig(os.path.join(cv_ff_fig_path, "FF-P{}.pdf".format(nn_stim)))
    
    os.chdir(cwd)
