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
        
    def get_trial_times(self):
        
        trial_duration = self.Ttrans + self.Tblank + self.Tstim
        
        self.st_tr_time = np.arange(self.Ntrials)*trial_duration
        self.end_tr_time = np.arange(1, self.Ntrials+1)*trial_duration
        
    def get_ind_fr(self, Id, duration, is_ms=True):
        
        if is_ms:
            c_factor = 1000
        else:
            c_factor = 1
        
        fr = np.histogram(Id, self.Nall)[0]/duration*c_factor
        
        fr_e, fr_i = fr[:self.NE], fr[self.NE:]
        
        return fr_e, fr_i
    
    def smooth(self, signal, win_size=20, kernel='rect'):
        
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
            
            e, i = self.get_ind_fr(sel_id, self.Tblank, is_ms=True)   
            
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
            
    def get_fr_diff(self, pert_val):
        
        self.base_exc, self.base_inh = self.get_fr(pert_val,
                                                   [self.Ttrans,
                                                    self.Ttrans+self.Tblank])
        self.stim_exc, self.stim_inh = self.get_fr(pert_val,
                                                   [self.Ttrans+self.Tblank, 
                                                    self.Ttrans+self.Tblank+self.Tstim])
        
        self.diff_exc = self.stim_exc - self.base_exc
        self.diff_inh = self.stim_inh - self.base_inh
        
        self.diff_exc_m = self.diff_exc.mean(axis=1)
        self.diff_inh_m = self.diff_inh.mean(axis=1)
        
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
        
    
cwd = os.getcwd()
fig_path = os.path.join(cwd, fig_dir+sim_suffix)
        
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
            
            a_r, a_c = ii//3, ii%3
            
            simdata_obj.get_fr_diff(nn_stim)
            simdata_obj.get_indegree()
            simdata_obj.plot_indeg_frdiff(ax[a_r, a_c])
            simdata_obj.plot_indeg_frdiff_e(ax_e[a_r, a_c])
            
            simdata_obj.plot_frdiff_dist(ax_dist[a_r, a_c])
            
            simdata_obj.plot_fr_dist(ax_base[a_r, a_c])
            
            simdata_obj.plot_inpfr_frdiff(ax_i_fr[a_r, a_c])
            
            simdata_obj.plot_inpfr_frdiff_e(ax_e_fr[a_r, a_c])
            simdata_obj.plot_box_frdiff(ax_box[:, ij2], nn_stim)
            
            simdata_obj.get_avg_frs(nn_stim)
            simdata_obj.concat_avg_frs_perts(ii)
            
            ax[a_r, a_c].set_title('P={}'.format(nn_stim))
            ax_dist[a_r, a_c].set_title('P={}'.format(nn_stim))
            ax_base[a_r, a_c].set_title('P={}'.format(nn_stim))
            
        
        ax_box[0, ij2].set_title('Bi={}'.format(Bi))
        
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

os.chdir(cwd)