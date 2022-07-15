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
import searchParams; reload(searchParams); from searchParams import *
from raster import raster

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
    cwd = os.getcwd()
    fig_path = os.path.join(fig_dir, fig_dir+fig_initial+sim_suffix, fig_ca)
    os.makedirs(fig_path, exist_ok=True)
    '''
    Analysis directories
    '''
    raster_dir_tr = "Raster-by-A-Trial"
    if fig_ca == 'ca3':
        BE = np.random.choice(Be_rng, 1, replace=False)
    else:
        BE = Be_rng
    for ij1, Be in enumerate(BE):
        for ij2, Bi in enumerate(Bi_rng):
            os.chdir(os.path.join(data_dir, res_dir + sim_suffix))
            print('Directory {}\n'.format(os.path.join(data_dir, res_dir + sim_suffix)))
            sim_name = file_name.format(Be, Bi)
            print('Reading {} ...\n'.format(sim_name))
            fig_raster_tr, ax_raster_tr = plt.subplots(ncols=1, nrows=nn_stim_rng.size,
                                                       sharex=True, sharey=True,
                                                       figsize=(6, 6))
            raster_obj = raster(file_name, Be, Bi)
            raster_tr_fig = raster_obj.create_fig_subdir(fig_path, raster_dir_tr)
            ax_raster_tr[int(nn_stim_rng.size/2)].set_ylabel('Neuron ID')
            ax_raster_tr[-1].set_xlabel('Time (ms)')
            raster_obj.plot_by_trial(ax_raster_tr)
            fig_raster_tr.savefig(os.path.join(raster_tr_fig, "Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
                        format="pdf")
            plt.close(fig_raster_tr)
    
if __name__=='__main__':

    if len(sys.argv) == 1:
        job_id = 0; num_jobs = 1
    else:
        job_id = int(sys.argv[1])
        num_jobs = int(sys.argv[2])
        
    file_names = ['sim_res_Be{:.2f}_Bi{:.2f}', 'sim_res_ca3_Be{:.2f}_Bi{:.2f}']
    
    EE_probchg_comb = EE_probchg_comb.flatten()[job_id::num_jobs]
    EI_probchg_comb = EI_probchg_comb.flatten()[job_id::num_jobs]
    II_condchg_comb = II_condchg_comb.flatten()[job_id::num_jobs]
    #fr_chg_comb = fr_chg_comb.flatten()[job_id::num_jobs]
    E_extra_comb = E_extra_comb.flatten()[job_id::num_jobs]
    bkg_chg_comb = bkg_chg_comb.flatten()[job_id::num_jobs]
    # CA3_CP_comb = CA3_CP_comb.flatten()[job_id::num_jobs]
    
    for ij1 in range(EE_probchg_comb.size):
        '''
        sim_suffix = "-EIeqpert-bkgfac{:.2f}-Epertfac{:.1f}-longersim-HEEcond-EE_probchg{:.2f}-EI_probchg{:.2f}".format(bkg_chg_comb[ij1],
                                                                 E_extra_comb[ij1],
                                                                 EE_probchg_comb[ij1],
                                                                 EI_probchg_comb[ij1])
        '''
        
        for file_name in file_names:  
            print('sim_suf={}'.format(file_name))        
            if 'ca3' in file_name:
                fig_ca = 'ca3'
            else:
                fig_ca = 'ca1'
        
            # sim_suffix_comp = sim_suffix.format(CA3_CP_comb[ij1], extra_bkg_e, E3E1_cond_chg, Bi_ca3, Be_ca3, r_bkg_ca1, E_extra_comb[ij1], EE_probchg_comb[ij1], EI_probchg_comb[ij1])
            sim_suffix_comp = sim_suffix.format(extra_bkg_e, E3E1_cond_chg, Bi_ca3, Be_ca3, r_bkg_ca1, E_extra_comb[ij1], EE_probchg_comb[ij1], EI_probchg_comb[ij1])

            run_for_each_parset(sim_suffix_comp, file_name, fig_ca)
