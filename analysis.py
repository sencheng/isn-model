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

def smooth(sig, kernel="rect", win_width=10):
        filter_ = np.ones(win_width)/win_width
        return np.convolve(sig, filter_, mode="same")

def std_chunk(signal, time, bin_width):
    time_vec = np.arange(time[0], time[-1]+bin_width, bin_width)
    std = np.zeros(time_vec.size-1)
    for t_i in range(time_vec.size-1):
        sel_ind = (time<time_vec[t_i+1]) & (time>=time_vec[t_i])
        std[t_i] = np.std(signal[sel_ind])
    time_vec = (time_vec[:-1] + time_vec[1:])/2
    return time_vec, std
    
def _get_fr(ids, times, interval):
        ids_sel = ids[(times>=interval[0]) & (times<interval[1])]
        return np.histogram(ids_sel, np.arange(1, N+2))[0]/np.diff(interval)*1000

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
        
        fl = open(fl_path, 'rb'); self.data = pickle.load(fl); fl.close()
        self.trans_interval = np.array([0, Ttrans])
        self.base_interval = self.trans_interval + Tblank
        self.stim_interval = self.base_interval + Tstim
        self.evoke_interval = self.stim_interval + Tblank
        self.ids_included = np.arange(1, N+1)
        self.neuron_ids = np.arange(1, N+1)
        self.fr_exc = {}
        self.fr_inh = {}
        self.ev_resp_exc = {}
        self.ev_resp_inh = {}
        self.Ntrials = Ntrials*C_rng.size
        self.mem_pots = {}
        self.conds = {}
        self.color_dict = {'exc': 'red', 'inh': 'blue'}
        if C_rng.size == 1:
            self.trials = np.arange((C_rng[0]-1)*Ntrials, C_rng[0]*Ntrials, 1)
        else:
            self.trials = np.arange((C_rng[0]-1)*Ntrials, C_rng[-1]*Ntrials, 1)
            
    def get_fr(self, pert, interval):
        fr = np.zeros((N, self.Ntrials))
        for tr_ind, tr in enumerate(self.trials):
            ids = self.data[pert][2][tr]['senders']
            spk_times = self.data[pert][2][tr]['times']
            fr[:, tr_ind] = _get_fr(ids, spk_times, interval)
        return fr[:NE, :], fr[NE:, :]
    
    def select_neurons(self, pert, interval, min_fr):
        fr = np.zeros((N, self.Ntrials))
        for tr_ind, tr in enumerate(self.trials):
            ids = self.data[pert][2][tr]['senders']
            spk_times = self.data[pert][2][tr]['times']
            fr[:, tr_ind] = _get_fr(ids, spk_times, interval)
        fr = fr.mean(axis=1)
        sel_ids = np.where(fr >= min_fr)[0] + 1
        if self.ids_included.size < N:
            print("\n######\n You have done a selection earlier!\n######\n")
            ids_included = []
            for id_ in sel_ids:
                if id_ in self.ids_included:
                    ids_included.append(id_)
            self.ids_included = np.array(ids_included)
        else:
            self.ids_included = sel_ids
            
    def _get_gain_pert(self, pert, interval):
        fr = np.zeros((N, self.Ntrials))
        for tr_ind, tr in enumerate(self.trials):
            ids = self.data[pert][2][tr]['senders']
            spk_times = self.data[pert][2][tr]['times']
            fr[:, tr_ind] = _get_fr(ids, spk_times, interval)
        return fr.mean(axis=1)
            
    def _get_pop_fr_pert(self, pert, dt):
        ids = []
        spk_times = []
        for tr in self.trials:
            ids.append(self.data[pert][2][tr]['senders'])
            spk_times.append(self.data[pert][2][tr]['times'])
        ids = np.concatenate(ids)
        spk_times = np.concatenate(spk_times)
        time_edges = np.arange(0, self.evoke_interval[1]+dt+dt/10, dt)
        time_vec = (time_edges[1:] + time_edges[0:-1])/2
        pop_fr_exc = np.zeros(time_vec.size)
        pop_fr_inh = np.zeros_like(pop_fr_exc)
        for inc_id in self.ids_included:
            if inc_id > NE:
                sel_spk_times = spk_times[(ids==inc_id)]
                pop_fr_inh += np.histogram(sel_spk_times, time_edges)[0]
            else:
                sel_spk_times = spk_times[(ids==inc_id)]
                pop_fr_exc += np.histogram(sel_spk_times, time_edges)[0]
        pop_fr_inh = pop_fr_inh/dt*1000/np.sum(self.ids_included>NE)/self.Ntrials
        pop_fr_exc = pop_fr_exc/dt*1000/np.sum(self.ids_included<=NE)/self.Ntrials
        return smooth(pop_fr_exc), smooth(pop_fr_inh), time_vec
    
    def plot_activated_vs_evoked_rel(self, ax, pert=NI):
        
        print("\nPlotting correlation ...\n")
        fr_base = self._get_gain_pert(pert, self.base_interval)
        fr_stim = self._get_gain_pert(pert, self.stim_interval)
        fr_evok = self._get_gain_pert(pert, self.evoke_interval)
        ev_int = np.arange(NE-int(NE*ev_prop), NE)
        nonev_int = np.arange(0, NE-int(NE*ev_prop))
        ax[0].scatter(fr_stim[ev_int]-fr_base[ev_int], fr_evok[ev_int]-fr_stim[ev_int], color='red')
        ax[1].scatter(fr_stim[nonev_int]-fr_base[nonev_int], fr_evok[nonev_int]-fr_stim[nonev_int],
                      color='red', label='excitatory')
        ax[1].scatter(fr_stim[NE:]-fr_base[NE:], fr_evok[NE:]-fr_stim[NE:],
                      color='blue', label='inhibitory')
        ax[0].set_ylabel(r'$f_{evoked}-f_{activated}$')
        ax[0].set_xlabel(r'$f_{activated}-f_{spontaneous}$')
        ax[1].set_xlabel(r'$f_{activated}-f_{spontaneous}$')
        ax[1].legend()
    
    def get_pop_fr(self, dt=10.):
        for pert in nn_stim_rng:
            print("\nComputing population firing rate for pert={:.0f}%".format(pert/NI*100))
            e_f, i_f, t = self._get_pop_fr_pert(pert, dt)
            self.fr_exc[pert] = e_f
            self.fr_inh[pert] = i_f
        self.fr_exc['time'] = t
        self.fr_inh['time'] = t
        return self.fr_exc, self.fr_inh
        
    def get_gain(self):
        for pert in nn_stim_rng:
            fr_base = self._get_gain_pert(pert, self.stim_interval)
            fr_evoke = self._get_gain_pert(pert, self.evoke_interval)
            fr_diff = fr_evoke - fr_base
            self.ev_resp_exc[pert] = fr_diff[self.ids_included[self.ids_included<=NE]-1]
            self.ev_resp_inh[pert] = fr_diff[self.ids_included[self.ids_included>NE]-1]
        return self.ev_resp_exc, self.ev_resp_inh
            
    def plot_pop_fr(self, ax):
        
        print("\nPlotting the population firing rates\n")
        if isinstance(ax, np.ndarray):
            for ax_ind, ax_ in enumerate(ax):
                ax_.plot(self.fr_exc['time'],
                         self.fr_exc[nn_stim_rng[ax_ind]],
                         color='red', linewidth=2, label='Exc.')
                ax_.plot(self.fr_inh['time'],
                         self.fr_inh[nn_stim_rng[ax_ind]],
                         color='blue', linewidth=2, label='Inh.')
                ax_.set_xlabel('Time (ms)')
                if nn_stim_rng[ax_ind] == 0:
                    title = 'control'
                else:
                    title = 'Pert={:.0f}%'.format(nn_stim_rng[ax_ind]/NI*100)
                ax_.set_title(title)
            ax[0].set_ylabel('Firing rate (spk/s)')
            ax[-1].legend()
        else:
            ax.plot(self.fr_exc['time'],
                    self.fr_exc[nn_stim_rng[0]],
                     color='red', linewidth=2, label='Exc.')
            ax.plot(self.fr_inh['time'],
                     self.fr_inh[nn_stim_rng[0]],
                     color='blue', linewidth=2, label='Inh.')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing rate (spk/s)')
            ax.legend()
    
    def _get_tr_avg_voltages(self, pert):
        
        ids = self.data[pert][-1][0]['senders']
        u_ids = np.unique(ids)
        times = np.unique(self.data[pert][-1][0]['times'])
        voltages = np.zeros((u_ids.size, times.size))
        for tr_ind, tr in enumerate(self.trials):
            for i_id, id_ in enumerate(u_ids):
                sel_id = self.data[pert][-1][tr]['senders'] == id_
                voltages[i_id, :] += self.data[pert][-1][tr]['V_m'][sel_id]
        return u_ids, voltages/self.Ntrials
        
    def _get_voltage_pert(self):
        
        times = np.unique(self.data[0][-1][0]['times'])
        for pert in nn_stim_rng:
            print("\nExtracting the voltages for data: pert={:.0f}%".format(pert/NI*100))
            ids, mem_pot_tr = self._get_tr_avg_voltages(pert)
            change_boundries = np.where(np.diff(ids)>1)[0]+1
            mem_pot_pert = {'non': {}, 'act_only': {}, 'ev_only': {}, 'both': {}}
            if pert == 0:
                mem_pot_pert['non'] = mem_pot_tr[:change_boundries[0]]
                mem_pot_pert['ev_only'] = mem_pot_tr[change_boundries[0]:]
                mem_pot_pert['act_only'] = np.nan*np.ones(mem_pot_tr.shape[1])
                mem_pot_pert['both'] = np.nan*np.ones(mem_pot_tr.shape[1])
            elif pert < NI:
                mem_pot_pert['non'] = mem_pot_tr[change_boundries[0]:change_boundries[1]]
                mem_pot_pert['ev_only'] = mem_pot_tr[change_boundries[0]:]
                mem_pot_pert['act_only'] = mem_pot_tr[change_boundries[1]:]
                mem_pot_pert['both'] = np.nan*np.ones(mem_pot_tr.shape[1])
            else:
                mem_pot_pert['non'] = np.nan*np.ones(mem_pot_tr.shape[1])
                mem_pot_pert['ev_only'] = np.nan*np.ones(mem_pot_tr.shape[1])
                mem_pot_pert['act_only'] = mem_pot_tr[change_boundries[0]:]
                mem_pot_pert['both'] = mem_pot_tr[change_boundries[0]:]
            self.mem_pots[pert] = mem_pot_pert
        self.mem_pots['times'] = times
        
    def plot_avg_mem_pots(self, ax, plot_stds=True, ms_bin_width=100.0):
        
        if plot_stds:
            ax2 = np.empty_like(ax)
            for row in range(ax.shape[0]):
                for col in range(ax.shape[1]):
                    ax2[row, col] = ax[row, col].twinx()
        self._get_voltage_pert()
        for p_i, pert in enumerate(nn_stim_rng):
            print("\nPlotting the average voltages for pert={:.0f}".format(pert/NI*100))
            for k_i, key in enumerate(self.mem_pots[pert].keys()):
                voltages = self.mem_pots[pert][key]
                if len(voltages.shape)>1:
                    voltages = voltages.mean(axis=0)
                ax[p_i, k_i].plot(self.mem_pots['times'], voltages, color='black', linewidth=1)
                time, std = std_chunk(voltages, self.mem_pots['times'], ms_bin_width)
                ax2[p_i, k_i].plot(time, std, color='green', linewidth=1)
                if p_i == 0:
                    ax[p_i, k_i].set_title(key)
                if p_i == nn_stim_rng.size-1:
                    ax[p_i, k_i].set_xlabel('Times (ms)')
                if k_i==0:
                    ax[p_i, k_i].set_ylabel('Membrane potential (mV)')
                if k_i == 3:
                    ax2[p_i, k_i].set_ylabel('Std of Membrane potential')

    def plot_ind_mem_pots(self, ax):
    
        self._get_voltage_pert()
        for p_i, pert in enumerate(nn_stim_rng):
            print("\nPlotting individual membrane potentials for pert={:.0f}%".format(pert/NI*100))
            for k_i, key in enumerate(self.mem_pots[pert].keys()):
                voltages = self.mem_pots[pert][key]
                ax[p_i, k_i].plot(self.mem_pots['times'], voltages.T, color='grey', linewidth=0.5)
                if p_i == 0:
                    ax[p_i, k_i].set_title(key)
                if p_i == nn_stim_rng.size-1:
                    ax[p_i, k_i].set_xlabel('Times (ms)')
                if k_i==0:
                    ax[p_i, k_i].set_ylabel('Membrane potential (mV)')

    def plot_trbytr_ind_mem_pots(self, ax):
        
        times = np.unique(self.data[0][-1][0]['times'])
        sel_tr = np.random.choice(np.arange(self.Ntrials))
        for p_i, pert in enumerate(nn_stim_rng):
            print("\nPlotting individual membrane potentials for individual trials for pert={:.0f}%".format(pert/NI*100))
            ids = self.data[pert][-1][0]['senders']
            u_ids = np.unique(ids)
            if pert == 0:
                ids_to_plot = [u_ids[0], u_ids[-1]]
            elif pert < NI:
                ids_to_plot = [u_ids[0], u_ids[rec_from_n_neurons], u_ids[-1]]
            else:
                ids_to_plot = [u_ids[0], u_ids[-1]]
            for id_i, id_ in enumerate(ids_to_plot):
                chunk = ids==id_
                ax[p_i, id_i].plot(times, self.data[pert][-1][sel_tr]['V_m'][chunk], linewidth=2)
                if p_i == nn_stim_rng.size-1:
                    ax[p_i, id_i].set_xlabel('Times (ms)')
                if id_i==0:
                    ax[p_i, id_i].set_ylabel('Membrane potential (mV)')

    def _get_tr_avg_cond(self, pert):
        
        ids = self.data[pert][3][0]['senders']
        u_ids = np.unique(ids)
        times = np.unique(self.data[pert][3][0]['times'])
        cond_exc = np.zeros((u_ids.size, times.size))
        cond_inh = np.zeros((u_ids.size, times.size))
        for tr_ind, tr in enumerate(self.trials):
            for i_id, id_ in enumerate(u_ids):
                sel_id = self.data[pert][3][tr]['senders'] == id_
                cond_exc[i_id, :] += self.data[pert][3][tr]['g_ex'][sel_id]
                cond_inh[i_id, :] += self.data[pert][3][tr]['g_in'][sel_id]
        return u_ids, {'exc': cond_exc/self.Ntrials, 'inh': cond_inh/self.Ntrials}
        
    def _get_cond_pert(self):
        
        times = np.unique(self.data[0][3][0]['times'])
        for pert in nn_stim_rng:
            print("\nExtracting the conductances for data: pert={:.0f}%".format(pert/NI*100))
            ids, cond_tr = self._get_tr_avg_cond(pert)
            change_boundries = np.where(np.diff(ids)>1)[0]+1
            cond_pert = {'exc': {'non': {}, 'act_only': {}, 'ev_only': {}, 'both': {}},
                         'inh': {'non': {}, 'act_only': {}, 'ev_only': {}, 'both': {}}}
            for ex_in in cond_tr.keys():
                if pert == 0:
                    cond_pert[ex_in]['non'] = cond_tr[ex_in][:change_boundries[0]]
                    cond_pert[ex_in]['ev_only'] = cond_tr[ex_in][change_boundries[0]:]
                    cond_pert[ex_in]['act_only'] = np.nan*np.ones(cond_tr[ex_in].shape[1])
                    cond_pert[ex_in]['both'] = np.nan*np.ones(cond_tr[ex_in].shape[1])
                elif pert < NI:
                    cond_pert[ex_in]['non'] = cond_tr[ex_in][change_boundries[0]:change_boundries[1]]
                    cond_pert[ex_in]['ev_only'] = cond_tr[ex_in][change_boundries[0]:]
                    cond_pert[ex_in]['act_only'] = cond_tr[ex_in][change_boundries[1]:]
                    cond_pert[ex_in]['both'] = np.nan*np.ones(cond_tr[ex_in].shape[1])
                else:
                    cond_pert[ex_in]['non'] = np.nan*np.ones(cond_tr[ex_in].shape[1])
                    cond_pert[ex_in]['ev_only'] = np.nan*np.ones(cond_tr[ex_in].shape[1])
                    cond_pert[ex_in]['act_only'] = cond_tr[ex_in][change_boundries[0]:]
                    cond_pert[ex_in]['both'] = cond_tr[ex_in][change_boundries[0]:]
            self.conds[pert] = cond_pert
        self.conds['times'] = times
        
    def plot_avg_conds(self, ax):
        
        self._get_cond_pert()
        for p_i, pert in enumerate(nn_stim_rng):
            print("\nPlotting the average conductances for pert={:.0f}%".format(pert/NI*100))
            for ei_i, ex_in in enumerate(self.conds[pert].keys()):
                for k_i, key in enumerate(self.conds[pert][ex_in].keys()):
                    conds = self.conds[pert][ex_in][key]
                    if len(conds.shape)>1:
                        conds = conds.mean(axis=0)
                    ax[p_i, k_i].plot(self.conds['times'], conds,
                                      color=self.color_dict[ex_in],
                                      label=ex_in,
                                      linewidth=1)
                    if p_i == 0:
                        ax[p_i, k_i].set_title(key)
                    if p_i == nn_stim_rng.size-1:
                        ax[p_i, k_i].set_xlabel('Times (ms)')
                    if k_i==0:
                        ax[p_i, k_i].set_ylabel('Conductance (nS)')

    def plot_ind_conds(self, ax):
    
        self._get_cond_pert()
        for p_i, pert in enumerate(nn_stim_rng):
            print("\nPlotting individual conductances for pert={:.0f}%".format(pert/NI*100))
            for ei_i, ex_in in enumerate(self.conds[pert].keys()):
                for k_i, key in enumerate(self.conds[pert][ex_in].keys()):
                    conds = self.conds[pert][ex_in][key]
                    ax[p_i, k_i].plot(self.conds['times'], conds.T,
                                      color=self.color_dict[ex_in],
                                      alpha=0.5, linewidth=0.5)
                    if p_i == 0:
                        ax[p_i, k_i].set_title(key)
                    if p_i == nn_stim_rng.size-1:
                        ax[p_i, k_i].set_xlabel('Times (ms)')
                    if k_i==0:
                        ax[p_i, k_i].set_ylabel('Conductances (nS)')

    def plot_trbytr_ind_conds(self, ax):
        
        times = np.unique(self.data[0][3][0]['times'])
        sel_tr = np.random.choice(np.arange(self.Ntrials))
        for p_i, pert in enumerate(nn_stim_rng):
            print("\nPlotting individual conductances for individual trials for pert={:.0f}%".format(pert/NI*100))
            ids = self.data[pert][3][0]['senders']
            u_ids = np.unique(ids)
            if pert == 0:
                ids_to_plot = [u_ids[0], u_ids[-1]]
            elif pert < NI:
                ids_to_plot = [u_ids[0], u_ids[rec_from_n_neurons], u_ids[-1]]
            else:
                ids_to_plot = [u_ids[0], u_ids[-1]]
            for id_i, id_ in enumerate(ids_to_plot):
                chunk = ids==id_
                ax[p_i, id_i].plot(times, self.data[pert][3][sel_tr]['g_ex'][chunk],
                                   color=self.color_dict['exc'], label='exc',
                                   linewidth=2)
                ax[p_i, id_i].plot(times, self.data[pert][3][sel_tr]['g_in'][chunk],
                                   color=self.color_dict['inh'], label='inh',    
                                   linewidth=2)
                if p_i == nn_stim_rng.size-1:
                    ax[p_i, id_i].set_xlabel('Times (ms)')
                if id_i==0:
                    ax[p_i, id_i].set_ylabel('Conductances (nS)')
