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
    
    def __init__(self, fl_path):
        
        fl = open(fl_path, 'rb'); sim_res = pickle.load(fl); fl.close()
        
        self.Ntrials = sim_res['Ntrials']
        self.NE = sim_res['NE']
        self.NI = sim_res['NI']
        self.Nall = sim_res['N']
        
        self.Ttrans = sim_res['Ttrans']
        self.Tstim  = sim_res['Tstim']
        self.Tblank = sim_res['Tblank']
        self.Texp   = self.Ttrans + self.Tstim + self.Tblank
        
        self.w_etoe = sim_res['W_EtoE']
        self.w_etoi = sim_res['W_EtoI']
        self.w_itoe = sim_res['W_ItoE']        
        self.w_itoi = sim_res['W_ItoI']
        del sim_res['W_EtoE'], sim_res['W_EtoI'], sim_res['W_ItoE'], sim_res['W_ItoI']
        
        self.sim_res = sim_res
        
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
        
    def get_trial_times(self):
        
        '''
        Trials appear in a single simulation session one after another. This
        method extract starting (attr: st_tr_time) and finishing time
        (attr: end_tr_time) of each trial.
        '''
        
        self.st_tr_time = np.arange(self.Ntrials)*self.Texp
        self.end_tr_time = np.arange(1, self.Ntrials+1)*self.Texp
        
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
        
        fr_e = np.histogram(exc_times, edges)[0]/exc_num*c_factor
        fr_i = np.histogram(inh_times, edges)[0]/inh_num*c_factor
        
        fr_e = self.smooth(fr_e)
        fr_i = self.smooth(fr_i)
        
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
        
        spk_times = self.sim_res[pert_val][2]['times']
        spk_ids   = self.sim_res[pert_val][2]['senders']
        
        e_cv = np.zeros(self.Ntrials)
        i_cv, e_ff, i_ff = np.zeros_like(e_cv), np.zeros_like(e_cv), np.zeros_like(e_cv)
        
        for tr in range(self.Ntrials):
            
            # sel_sp_t = spk_times[(spk_times >= self.st_tr_time[i]+interval[0]) & 
            #                      (spk_times <  self.st_tr_time[i]+interval[1])]
            sel_id   = spk_ids[(spk_times >= self.st_tr_time[tr]+interval[0]) &
                               (spk_times <  self.st_tr_time[tr]+interval[1])]
            
            spk_time = spk_times[(spk_times >= self.st_tr_time[tr]+interval[0]) &
                                 (spk_times <  self.st_tr_time[tr]+interval[1])]
            
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
        
    def get_fr(self, pert_val, interval):
        
        if not hasattr(self, 'st_tr_time'):
            self.get_trial_times()
        
        fr_inh = np.zeros((self.NI, self.Ntrials))
        fr_exc = np.zeros((self.NE, self.Ntrials))
        
        spk_times = self.sim_res[pert_val][2]['times']
        spk_ids   = self.sim_res[pert_val][2]['senders']
        
        for tr in range(self.Ntrials):
            
            # sel_sp_t = spk_times[(spk_times >= self.st_tr_time[i]+interval[0]) & 
            #                      (spk_times <  self.st_tr_time[i]+interval[1])]
            sel_id   = spk_ids[(spk_times >= self.st_tr_time[tr]+interval[0]) &
                               (spk_times <  self.st_tr_time[tr]+interval[1])]
            
            e, i = self.get_ind_fr(sel_id, np.diff(interval), is_ms=True)   
            
            fr_exc[:, tr] = e
            fr_inh[:, tr] = i
            
        return fr_exc, fr_inh
    
    def get_fr_time(self, pert_val, interval):
        
        if not hasattr(self, 'st_tr_time'):
            self.get_trial_times()
            
        bin_size = 1
        T_edges_def = np.arange(interval[0],
                                interval[1],
                                bin_size)
        
        fr_inh = np.zeros((self.Ntrials, T_edges_def.size-1))
        fr_exc = np.zeros((self.Ntrials, T_edges_def.size-1))
        
        spk_times = self.sim_res[pert_val][2]['times']
        spk_ids   = self.sim_res[pert_val][2]['senders']
        
        for tr in range(self.Ntrials):
            
            T_edges = T_edges_def + self.st_tr_time[tr]
            
            # sel_sp_t = spk_times[(spk_times >= self.st_tr_time[i]+interval[0]) & 
            #                      (spk_times <  self.st_tr_time[i]+interval[1])]
            sel_id   = spk_ids[(spk_times >= self.st_tr_time[tr]+interval[0]) &
                               (spk_times <  self.st_tr_time[tr]+interval[1])]
            
            sel_T    = spk_times[(spk_times >= self.st_tr_time[tr]+interval[0]) &
                                 (spk_times <  self.st_tr_time[tr]+interval[1])]
            
            
            
            e, i = self.get_pop_fr(sel_id, sel_T, T_edges, is_ms=True)   
            
            fr_exc[tr, :] = e
            fr_inh[tr, :] = i
            
        return fr_exc, fr_inh, T_edges[0:-1]+bin_size/2
            
    def get_fr_diff(self, pert_val, exclude_inactives=True):
        
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
        
        self.trans_cv[0, :], self.trans_cv[1, :], self.trans_ff[0, :], self.trans_ff[1, :] = \
        self.get_cv(pert_val, [0, self.Ttrans])
        
        self.base_cv[0, :], self.base_cv[1, :], self.base_ff[0, :], self.base_ff[1, :] = \
        self.get_cv(pert_val, [self.Ttrans, self.Ttrans+self.Tblank])
        
        self.stim_cv[0, :], self.stim_cv[1, :], self.stim_ff[0, :], self.stim_ff[1, :] = \
        self.get_cv(pert_val, [self.Ttrans+self.Tblank, self.Ttrans+self.Tblank+self.Tstim])
        
        self.diff_exc = self.stim_exc - self.base_exc
        self.diff_inh = self.stim_inh - self.base_inh
        
        self.diff_exc_m = self.diff_exc.mean(axis=1)
        self.diff_inh_m = self.diff_inh.mean(axis=1)
        
        if exclude_inactives:
            
            self.diff_exc = self.diff_exc[(self.base_exc!=0) | (self.stim_exc!=0)]
            self.diff_inh = self.diff_inh[(self.base_inh!=0) | (self.stim_inh!=0)]
            
            self.paradox_score = ((np.sum(self.diff_inh>0)/self.diff_inh.size -\
                                   np.sum(self.diff_exc>0)/self.diff_exc.size)*\
                                   np.sum(self.diff_inh>0)/self.diff_inh.size)
        
            self.diff_exc_m = self.diff_exc.mean()
            self.diff_inh_m = self.diff_inh.mean()
        
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
        ax.set_title('r={:.2f}-'.format(out.pvalue)+ax.get_title())
        
        
    def plot_indeg_frdiff(self, ax):
        
        ax.scatter(self.diff_inh_m, self.indeg_etoi, s=1)
        self.plot_reg_line(self.diff_inh_m, self.indeg_etoi, ax)
        
    def plot_indeg_frdiff_e(self, ax):
        
        ax.scatter(self.diff_exc_m, self.indeg_etoe, s=1)
        self.plot_reg_line(self.diff_exc_m, self.indeg_etoe, ax)
        
    def plot_inpfr_frdiff(self, ax):

        sum_fr = np.zeros_like(self.stim_inh)        
        w_etoi_bin = self.w_etoi>0
        
        for i in range(self.NI):
            
            sum_fr[i, :] = self.stim_exc[w_etoi_bin[:, i], :].mean(axis=0)
        
        ax.scatter(self.diff_inh.flatten(), sum_fr.flatten())
        self.plot_reg_line(self.diff_inh.flatten(), sum_fr.flatten(), ax)
        
    def plot_inpfr_frdiff_e(self, ax):

        sum_fr = np.zeros_like(self.stim_exc)        
        w_etoe_bin = self.w_etoe>0
        
        for i in range(self.NE):
            
            sum_fr[i, :] = self.stim_exc[w_etoe_bin[:, i], :].mean(axis=0)
        
        ax.scatter(self.diff_exc.flatten(), sum_fr.flatten())
        self.plot_reg_line(self.diff_exc.flatten(), sum_fr.flatten(), ax)
        
    def plot_frdiff_dist(self, ax, num_bins=20):
        
        edges = np.linspace(self.diff_inh.min(), self.diff_inh.max(), num_bins)
        ax.hist([self.diff_inh.flatten(),
                 self.diff_exc.flatten()], edges, color=['blue', 'red'])
        
    def plot_box_frdiff(self, ax, pert_val):
        
        ax[0].boxplot(self.diff_inh.flatten(), positions=[pert_val],
                      widths=[25], flierprops={'marker': '.'})
        
        ax[1].boxplot(self.diff_exc.flatten(), positions=[pert_val],
                      widths=[25], flierprops={'marker': '.'})
        
    def plot_fr_dist(self, ax, num_bins=20):
        
        _max = max(self.base_exc.max(), self.base_inh.max())
        _min = min(self.base_exc.min(), self.base_inh.min())
        
        edges = np.linspace(_min, _max, num_bins)
        
        ax.hist([self.base_exc.flatten(),
        
                 self.base_inh.flatten()], edges, color=['red', 'blue'])
        
    def create_fig_subdir(self, path, dir_name):
        
        dir_path = os.path.join(path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        
        return dir_path
    
    def plot_raster_tr(self, ids, times, ax):
        
        for i, e_id in enumerate(self.vis_E_ids):
            
            sel_spks = times[ids==e_id]
            
            ax.plot(sel_spks, (i+1)*np.ones_like(sel_spks),
                    color='red', marker='|', markersize=1, linestyle='')
            
        for i, i_id in enumerate(self.vis_I_ids):
            
            sel_spks = times[ids==i_id]
            
            ax.plot(sel_spks, (i+self.vis_E_ids.size+1)*np.ones_like(sel_spks),
                    color='blue', marker='|', markersize=1, linestyle='')
        
    def plot_raster(self, pert_val, ax, prop=0.05):
        
        if not hasattr(self, 'st_tr_time'):
            self.get_trial_times()
            
        spk_times = self.sim_res[pert_val][2]['times']
        spk_ids   = self.sim_res[pert_val][2]['senders']
        
        vis_NE = int(self.NE*prop)
        vis_NI = int(self.NI*prop)
        
        self.vis_E_ids = np.random.choice(np.unique(spk_ids[spk_ids<=self.NE]),
                                          vis_NE, replace=False)
        self.vis_I_ids = np.random.choice(np.unique(spk_ids[spk_ids>self.NE]),
                                          vis_NI, replace=False)
            
        for i in range(self.Ntrials):
            
            spk_t = spk_times[(spk_times>=self.st_tr_time[i]) & 
                              (spk_times<=self.end_tr_time[i])] - self.st_tr_time[i]
            
            spk_id = spk_ids[(spk_times>=self.st_tr_time[i]) & 
                             (spk_times<=self.end_tr_time[i])]
            
            self.plot_raster_tr(spk_id, spk_t, ax[i])
            
        ax[-1].set_xlabel("Time (ms)")
        ax[2].set_ylabel("Neuron ID")
    
cwd = os.getcwd()
fig_path = os.path.join(cwd, fig_dir+sim_suffix)
os.makedirs(fig_path, exist_ok=True)

cv_e = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 3))
cv_i = np.zeros_like(cv_e)

ff_e = np.zeros_like(cv_e)
ff_i = np.zeros_like(cv_e)

paradox_score = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size))

        
for ij1, Be in enumerate(Be_rng):
    
    fig_box, ax_box = plt.subplots(nrows=2, ncols=Bi_rng.size,
                                   sharex=True, sharey=True)
    
    for ij2, Bi in enumerate(Bi_rng):
        
        os.chdir(os.path.join(cwd, res_dir + sim_suffix))
        sim_name = 'sim_res_Be'+str(Be)+'_Bi'+str(Bi)
        print('Reading {} ...\n'.format(sim_name))
        # fl = open(sim_name, 'rb'); sim_res = pickle.load(fl); fl.close()
        
        fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        fig_e, ax_e = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        fig_base, ax_base = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        fig_dist, ax_dist = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        fig_i_fr, ax_i_fr = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        fig_e_fr, ax_e_fr = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        
        fig_avg_fr, ax_avg_fr = plt.subplots()
        
        simdata_obj = simdata(sim_name)
        
        ax[0, 0].set_ylabel('E to I')
        ax[1, 0].set_ylabel('E to I')
        
        ax_i_fr[0, 0].set_ylabel('Avg. fr. E to I')
        ax_i_fr[1, 0].set_ylabel('Avg. fr. E to I')
        
        ax_e_fr[0, 0].set_ylabel('Avg. fr. E to E')
        ax_e_fr[1, 0].set_ylabel('Avg. fr. E to E')
        
        ax_e[0, 0].set_ylabel('E to E')
        ax_e[1, 0].set_ylabel('E to E')
        
        ax[1, 1].set_xlabel('Firing rate change (Hz)')
        ax_e[1, 1].set_xlabel('Firing rate change (Hz)')
        ax_i_fr[1, 1].set_xlabel('Firing rate change (Hz)')
        ax_e_fr[1, 1].set_xlabel('Firing rate change (Hz)')
        
        ax_base[1, 1].set_xlabel('Firing rate (Hz)')
        
        ax_dist[1, 1].set_xlabel('Firing rate change (Hz)')
        
        for ii, nn_stim in enumerate(nn_stim_rng):
            
            fig_raster, ax_raster = plt.subplots(nrows=Ntrials, ncols=1,
                                                 sharex=True, sharey=True)
            
            a_r, a_c = ii//3, ii%3
            
            simdata_obj.get_fr_diff(nn_stim)
            # simdata_obj.get_indegree()
            # simdata_obj.plot_indeg_frdiff(ax[a_r, a_c])
            # simdata_obj.plot_indeg_frdiff_e(ax_e[a_r, a_c])
            
            simdata_obj.plot_frdiff_dist(ax_dist[a_r, a_c])
            
            paradox_score[ij1, ij2, ii] = simdata_obj.paradox_score
            
            simdata_obj.plot_fr_dist(ax_base[a_r, a_c])
            
            simdata_obj.plot_box_frdiff(ax_box[:, ij2], nn_stim)
            
            # simdata_obj.plot_inpfr_frdiff(ax_i_fr[a_r, a_c])
            
            # simdata_obj.plot_inpfr_frdiff_e(ax_e_fr[a_r, a_c])
            
            simdata_obj.get_avg_frs(nn_stim)
            simdata_obj.concat_avg_frs_perts(ii)
            
            
            cv_e[ij1, ij2, ii, 0] = simdata_obj.trans_cv[0, :].mean()
            cv_e[ij1, ij2, ii, 1] = simdata_obj.base_cv[0, :].mean()
            cv_e[ij1, ij2, ii, 2] = simdata_obj.stim_cv[0, :].mean()
            
            ff_e[ij1, ij2, ii, 0] = simdata_obj.trans_ff[0, :].mean()
            ff_e[ij1, ij2, ii, 1] = simdata_obj.base_ff[0, :].mean()
            ff_e[ij1, ij2, ii, 2] = simdata_obj.stim_ff[0, :].mean()
            
            cv_i[ij1, ij2, ii, 0] = simdata_obj.trans_cv[1, :].mean()
            cv_i[ij1, ij2, ii, 1] = simdata_obj.base_cv[1, :].mean()
            cv_i[ij1, ij2, ii, 2] = simdata_obj.stim_cv[1, :].mean()
            
            ff_i[ij1, ij2, ii, 0] = simdata_obj.trans_ff[1, :].mean()
            ff_i[ij1, ij2, ii, 1] = simdata_obj.base_ff[1, :].mean()
            ff_i[ij1, ij2, ii, 2] = simdata_obj.stim_ff[1, :].mean()
            
            # path_raster_fig = simdata_obj.create_fig_subdir(fig_path, "raster_dir")
            # simdata_obj.plot_raster(nn_stim, ax_raster)
            # fig_raster.savefig(os.path.join(path_raster_fig,
            #                                 "Be{}-Bi{}-P{}.png".format(Be, Bi, nn_stim)),
            #                    format="png")
            plt.close(fig_raster)
            
            ax[a_r, a_c].set_title('P={}'.format(nn_stim))
            ax_dist[a_r, a_c].set_title('P={}'.format(nn_stim))
            ax_base[a_r, a_c].set_title('P={}'.format(nn_stim))
            
        
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
        
        fig_e_fr.savefig(os.path.join(fig_path, "fr-fr-diff-dist-Be{}-Bi{}.pdf".format(Be, Bi)),
                         format="pdf")
        
        fig_i_fr.savefig(os.path.join(fig_path, "fre-fr-diff-dist-Be{}-Bi{}.pdf".format(Be, Bi)),
                         format="pdf")
        
        fig_dist.savefig(os.path.join(fig_path, "fr-diff-dist-Be{}-Bi{}.pdf".format(Be, Bi)),
                         format="pdf")
        
        fig_base.savefig(os.path.join(fig_path, "fr-base-dist-Be{}-Bi{}.pdf".format(Be, Bi)),
                         format="pdf")
        
        fig_avg_fr.savefig(os.path.join(fig_path, "avgfr-Be{}-Bi{}.pdf".format(Be, Bi)),
                           format="pdf")
                
        plt.close(fig)
        plt.close(fig_e)
        plt.close(fig_dist)
        plt.close(fig_base)
        plt.close(fig_i_fr)
        plt.close(fig_e_fr)
        plt.close(fig_avg_fr)
        
    fig_box.savefig(os.path.join(fig_path, "fr-diff-box-Be{}.pdf".format(Be)),
                     format="pdf")
    
    plt.close(fig_box)
    
cv_ff_fig_path = os.path.join(fig_path, "CV-FF")
os.makedirs(cv_ff_fig_path, exist_ok=True)

Ge = np.append(0, Be_rng)
Gi = np.append(0, Bi_rng)

fig_psc, ax_psc = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
bar_psc = fig_psc.add_axes([0.7, 0.15, 0.02, 0.3])
    
for ii, nn_stim in enumerate(nn_stim_rng):
    
    f = ax_psc[ii//3, ii%3].pcolormesh(Gi, Ge, paradox_score[:, :, ii],
                                       vmin=paradox_score.min(),
                                       vmax=paradox_score.max())
    ax_psc[ii//3, ii%3].set_title("pert={:.0f}%".format(nn_stim/NI*100))
    
ax_psc[1, 0].set_xlabel("Inh. Cond.")
ax_psc[1, 1].set_xlabel("Inh. Cond.")

ax_psc[0, 0].set_ylabel("Exc. Cond.")
ax_psc[1, 0].set_ylabel("Exc. Cond.")

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
