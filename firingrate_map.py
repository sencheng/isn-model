import numpy as np
import matplotlib.pyplot as plt
from imp import reload
import defaultParams; reload(defaultParams); from defaultParams import *
import searchParams; reload(searchParams); from searchParams import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from analysis import simdata
import pickle

def boxoff(ax):
    
    """
    Removes the top and right spines of the axes given as inputs, similar to
    boxoff function of MATLAB. Nothing is returned and it works through reference.
    
    Args:
        Axis or array of axes returned for example from plt.subplots().
    """
    if hasattr(ax, 'shape'):
        if len(ax.shape)>1:
            for i in range(ax.shape[0]):            
                for j in range(ax.shape[1]):
                    ax[i, j].spines['top'].set_visible(False)
                    ax[i, j].spines['right'].set_visible(False)
        else:
            for i in range(ax.shape[0]):
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
def to_square_plots(ax):
    
    """
    Make the aspect ratio of xy-axis of a given axes to one, so that they appear
    in square shape.
    
    Args:
        Axis or array of axes returned for example from plt.subplots().
    """
    
    if hasattr(ax, 'shape'):
        if len(ax.shape)>1:
            for i in range(ax.shape[0]):            
                for j in range(ax.shape[1]):
                    ratio = ax[i, j].get_data_ratio()
                    ax[i, j].set_aspect(1.0/ratio)
        else:
            for i in range(ax.shape[0]):
                ratio = ax[i].get_data_ratio()
                ax[i].set_aspect(1.0/ratio)
    else:
        ratio = ax.get_data_ratio()
        ax.set_aspect(1.0/ratio)

def create_fig_subdir(path, dir_name):
        
        dir_path = os.path.join(path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        
        return dir_path
            
def run_for_each_parset(sim_suffix, file_name, fig_ca):
    # cwd = os.getcwd()
    map_plots  = "FiringMaps"
    # os.chdir(os.path.join(data_dir, res_dir + sim_suffix))
    sim_name = file_name.format(Be_rng[0], Bi_rng[0])
    print('Reading {} ...\n'.format(sim_name))
    print(os.path.join(data_dir, res_dir + sim_suffix, sim_name))
    analyze = simdata(os.path.join(data_dir, res_dir + sim_suffix, sim_name))
    fr_exc, fr_inh = analyze.get_fr(nn_stim_rng[0], analyze.stim_interval)
    return [fr_exc.max(), fr_exc.mean(axis=1).max(), fr_exc.mean()],\
           [fr_inh.max(), fr_inh.mean(axis=1).max(), fr_inh.mean()]
           
def plot_all(exc, inh, x, y, ax, fig):
    
    for c in range(len(exc)):
        ax[0, c].set_aspect("equal")
        p = ax[0, c].pcolor(x, y, exc[c].reshape(x.size, y.size),
                            vmin=exc[c].min(), vmax=exc[c].max())
        axins = inset_axes(ax[0, c],
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax[0, c].transAxes,
                   borderpad=0)
        cbar = fig.colorbar(p, cax=axins)
        if c == len(exc)-1:
            cbar.set_label("Firing rate (spk/s)", rotation=270, labelpad=10)
        ax[0, c].set_yticks(y[::2])
        ax[1, c].set_aspect("equal")
        p = ax[1, c].pcolor(x, y, inh[c].reshape(x.size, y.size),
                            vmin=inh[c].min(), vmax=inh[c].max())
        axins = inset_axes(ax[1, c],
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax[1, c].transAxes,
                   borderpad=0)
        cbar = fig.colorbar(p, cax=axins)
        if c == len(exc)-1:
            cbar.set_label("Firing rate (spk/s)", rotation=270, labelpad=15)
            
def plot_one(exc, x, y, ax, fig):
    if np.unique(y).size > 1:
        ax.set_aspect("equal")
        p = ax.pcolor(x, y, exc.reshape(x.size, y.size),
                      vmin=exc.min(), vmax=exc.max())
        axins = inset_axes(ax,
                           width="5%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0)
        cbar = fig.colorbar(p, cax=axins)
        cbar.set_label("Firing rate (spk/s)", rotation=270, labelpad=10)
        ax.set_yticks(y[::2])
    
def plot_line(exc, x, y, ax, fig):
    # ax.set_aspect("equal")
    if np.unique(y).size == 1:
        ax.plot(x, exc)
        
    
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
    if (Be_rng[0]==0) & (Bi_rng[0]==0) & (Bi_ca3 < 0):
        file_suffix = '-w-inh'
        ca = 'ca3'
        file_name = file_names[0]
    elif (Bi_rng == 0) & (Bi_ca3 < 0):
        file_suffix = '-wo-inh'
        ca = 'ca1'
        file_name = file_names[1]
    elif Bi_ca3 == 0:
        file_suffix = '-wo-inh'
        ca = 'ca3'
        file_name = file_names[0]
    else:
        file_suffix = '-w-inh'
        ca = 'ca1'
        file_name = file_names[1]
    #file_suffix = file_suffix + '-{:.2f}'.format(EE_probchg_comb.max())
    for ij1 in range(EE_probchg_comb.size):
        # sim_suffix_comp = sim_suffix.format(CA3_CP_comb[ij1], extra_bkg_e, E3E1_cond_chg, Bi_ca3, Be_ca3, r_bkg_ca1, E_extra_comb[ij1], EE_probchg_comb[ij1], EI_probchg_comb[ij1])
        sim_suffix_comp = sim_suffix.format(extra_bkg_e, E3E1_cond_chg, Bi_ca3, Be_ca3, r_bkg_ca1, E_extra_comb[ij1], EE_probchg_comb[ij1], EI_probchg_comb[ij1])
        frs_e, frs_i = run_for_each_parset(sim_suffix_comp, file_name, ca)
        fr_max_e[ij1], fr_meanmax_e[ij1], fr_mean_e[ij1] = frs_e
        fr_max_i[ij1], fr_meanmax_i[ij1], fr_mean_i[ij1] = frs_i
    res_dict = {'E': {'grand_avg_fr': fr_mean_e, 'max_avg_fr': fr_meanmax_e, 'max_fr': fr_max_e},
                'I': {'grand_avg_fr': fr_mean_i, 'max_avg_fr': fr_meanmax_i, 'max_fr': fr_max_i },
                'ee_coef': EEconn_chg_factor, 'ei_coef': EIconn_chg_factor}
    print('writing to file ' + ca + file_suffix + '-{:.2f}'.format(EE_probchg_comb.max()))
    with open(ca + file_suffix + '-{:.2f}'.format(EE_probchg_comb.max()), 'wb') as res_data_fl:
        pickle.dump(res_dict, res_data_fl, pickle.HIGHEST_PROTOCOL)
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
    fig.tight_layout()
    plot_one(fr_mean_e, EEconn_chg_factor, EIconn_chg_factor, ax, fig)
    ax.set_xlabel(r"$J_{EE}$ coefficient")#ax.set_xlabel(r"$E\rightarrow E$ connection probability factor")
    ax.set_ylabel(r"$J_{EI}$ coefficient")#ax.set_ylabel(r"$I\rightarrow E$ connection probability factor")
    fig.savefig(fig_ca+'-wo-inh'+'.pdf', bbox_inches='tight')   
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
    fig.tight_layout()
    plot_line(fr_mean_e, EEconn_chg_factor, EIconn_chg_factor, ax, fig)
    ax.set_xlabel(r"$J_{EE}$ coefficient")
    ax.set_ylabel(r"Firing rate (spike/sec)")
    boxoff(ax)
    to_square_plots(ax)
    fig.savefig(fig_ca+'-wo-inh-line'+'.pdf', bbox_inches='tight')
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
        fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                               figsize=(6, 4),
                               gridspec_kw={'wspace': 0.4, 'hspace': 0.01})
        fig.tight_layout()
        plot_all((fr_max_e, fr_meanmax_e, fr_mean_e),
                 (fr_max_i, fr_meanmax_i, fr_mean_i),
                 EEconn_chg_factor, 
                 EIconn_chg_factor,
                 ax, fig)
        ax[-1, 1].set_xlabel(r"$E\rightarrow E$ connection probability factor")
        ax[0, 0].set_ylabel(r"$I\rightarrow E$ connection probability factor")
        fig.savefig(fig_ca+'-wo-inh-3analysis'+'.pdf')
    """
