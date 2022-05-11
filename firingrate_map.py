import numpy as np
import matplotlib.pyplot as plt
from imp import reload
import defaultParams; reload(defaultParams); from defaultParams import *
import searchParams; reload(searchParams); from searchParams import *
from analysis import simdata

def create_fig_subdir(path, dir_name):
        
        dir_path = os.path.join(path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        
        return dir_path

def boxoff(ax):
    
    """
    Removes the top and right spines of the axes given as inputs, similar to
    boxoff function of MATLAB. Nothing is returned and it works through reference.
    
    Args:
        Axis or array of axes returned for example from plt.subplots().
    """
    
    if len(ax.shape)>1:
        for i in range(ax.shape[0]):            
            for j in range(ax.shape[1]):
                ax[i, j].spines['top'].set_visible(False)
                ax[i, j].spines['right'].set_visible(False)
    else:
        for i in range(ax.shape[0]):
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)

def to_square_plots(ax):
    
    """
    Make the aspect ratio of xy-axis of a given axes to one, so that they appear
    in square shape.
    
    Args:
        Axis or array of axes returned for example from plt.subplots().
    """

    if len(ax.shape)>1:
        for i in range(ax.shape[0]):            
            for j in range(ax.shape[1]):
                ratio = ax[i, j].get_data_ratio()
                ax[i, j].set_aspect(1.0/ratio)
    else:
        for i in range(ax.shape[0]):
            ratio = ax[i].get_data_ratio()
            ax[i].set_aspect(1.0/ratio)
            
def run_for_each_parset(sim_suffix, file_name, fig_ca):
    # cwd = os.getcwd()
    map_plots  = "FiringMaps"
    # os.chdir(os.path.join(data_dir, res_dir + sim_suffix))
    sim_name = file_name.format(Be_rng[0], Bi_rng[0])
    print('Reading {} ...\n'.format(sim_name))
    analyze = simdata(os.path.join(data_dir, res_dir + sim_suffix, sim_name))
    fr_exc, fr_inh = analyze.get_fr(nn_stim_rng[0], analyze.base_interval)
    return [fr_exc.max(), fr_exc.mean(axis=1).max(), fr_exc.mean()],\
           [fr_inh.max(), fr_inh.mean(axis=1).max(), fr_inh.mean()]
           
def plot(exc, inh, x, y, ax, fig):
    
    for c in range(len(exc)):
        p = ax[0, c].pcolor(x, y, exc[c].reshape(x.size, y.size),
                            vmin=exc[c].min(), vmax=exc[c].max())
        fig.colorbar(p, ax=ax[0, c])
        p = ax[1, c].pcolor(x, y, inh[c].reshape(x.size, y.size),
                            vmin=inh[c].min(), vmax=inh[c].max())
        fig.colorbar(p, ax=ax[1, c])
if __name__=='__main__':
    
    fig_path = os.path.join(fig_dir, fig_initial+sim_suffix)
    os.makedirs(fig_path, exist_ok=True)
    file_names = ['sim_res_ca3_Be{:.2f}_Bi{:.2f}', 'sim_res_Be{:.2f}_Bi{:.2f}']
    EE_probchg_comb = EE_probchg_comb.flatten()
    EI_probchg_comb = EI_probchg_comb.flatten()
    II_condchg_comb = II_condchg_comb.flatten()
    #fr_chg_comb = fr_chg_comb.flatten()[job_id::num_jobs]
    E_extra_comb = E_extra_comb.flatten()
    bkg_chg_comb = bkg_chg_comb.flatten()
    # CA3_CP_comb = CA3_CP_comb.flatten()[job_id::num_jobs]
    fr_max_e = np.zeros(EE_probchg_comb.size)
    fr_max_i = np.zeros(EE_probchg_comb.size)
    fr_meanmax_e = np.zeros(EE_probchg_comb.size)
    fr_meanmax_i = np.zeros(EE_probchg_comb.size)
    fr_mean_e = np.zeros(EE_probchg_comb.size)
    fr_mean_i = np.zeros(EE_probchg_comb.size)
    for file_name in file_names:  
        print('sim_suf={}'.format(file_name))        
        if 'ca3' in file_name:
            fig_ca = 'ca3'
        else:
            fig_ca = 'ca1'
        for ij1 in range(EE_probchg_comb.size):
            # sim_suffix_comp = sim_suffix.format(CA3_CP_comb[ij1], extra_bkg_e, E3E1_cond_chg, Bi_ca3, Be_ca3, r_bkg_ca1, E_extra_comb[ij1], EE_probchg_comb[ij1], EI_probchg_comb[ij1])
            sim_suffix_comp = sim_suffix.format(extra_bkg_e, E3E1_cond_chg, Bi_ca3, Be_ca3, r_bkg_ca1, E_extra_comb[ij1], EE_probchg_comb[ij1], EI_probchg_comb[ij1])
            frs_e, frs_i = run_for_each_parset(sim_suffix_comp, file_name, fig_ca)
            fr_max_e[ij1], fr_meanmax_e[ij1], fr_mean_e[ij1] = frs_e
            fr_max_i[ij1], fr_meanmax_i[ij1], fr_mean_i[ij1] = frs_i
        fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        plot((fr_max_e, fr_meanmax_e, fr_mean_e),
             (fr_max_i, fr_meanmax_i, fr_mean_i),
             EEconn_chg_factor, 
             EIconn_chg_factor,
             ax, fig)