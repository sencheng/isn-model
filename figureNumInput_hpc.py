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
from figureNumInput import simdata, frchg_vs_EtoI,propposfrchg, frchg

def run_for_each_parset(sim_suffix, fig_ca):
    cwd = os.getcwd()
    
    cv_e = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 3))
    cv_i = np.zeros_like(cv_e)
    
    ff_e = np.zeros_like(cv_e)
    ff_i = np.zeros_like(cv_e)
    
    pos_prop = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 2))
    mean_fr  = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 2))
    
    paradox_score = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size))
    
    fig_path = os.path.join(cwd, fig_dir+sim_suffix, fig_ca)
    os.makedirs(fig_path, exist_ok=True)
    for ij1, Be in enumerate(Be_rng):
        
        fig_box, ax_box = plt.subplots(nrows=2, ncols=Bi_rng.size,
                                       sharex=True, sharey=True,
                                       figsize=(6, 6))
        
        fig_box_g, ax_box_g = plt.subplots(nrows=2, ncols=Bi_rng.size,
                                       sharex=True, sharey=True)
        
        
        
        for ij2, Bi in enumerate(Bi_rng):
        
            os.chdir(os.path.join(cwd, res_dir + sim_suffix))
            sim_name = sim_suf.format(Be, Bi)
            # sim_name = 'sim_res_Be{:.2f}_Bi{:.2f}'.format(Be, Bi)
            print('Reading {} ...\n'.format(sim_name))
            # fl = open(sim_name, 'rb'); sim_res = pickle.load(fl); fl.close()
            
            fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            fig_e, ax_e = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            fig_base, ax_base = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            fig_dist, ax_dist = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            fig_dist_g, ax_dist_g = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            fig_i_fr, ax_i_fr = plt.subplots(nrows=2, ncols=5, sharex=True, sharey='row')
            fig_e_fr, ax_e_fr = plt.subplots(nrows=2, ncols=5, sharex=True, sharey='row')
            fig_base_frdiff, ax_base_frdiff = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
            
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
                
                fig_raster, ax_raster = plt.subplots(nrows=Ntrials, ncols=1,
                                                     sharex=True, sharey=True)
                
                fig_raster_sep, ax_raster_sep = plt.subplots(nrows=2, ncols=2,
                                                             sharex=True)
                
                a_r, a_c = ii//3, ii%3
                
                simdata_obj.get_fr_diff(nn_stim)
                
                # simdata_obj.get_cond_diff(nn_stim)
                # simdata_obj.get_indegree()
                # simdata_obj.plot_indeg_frdiff(ax[a_r, a_c])
                # simdata_obj.plot_indeg_frdiff_e(ax_e[a_r, a_c])
                
                simdata_obj.plot_frdiff_dist(ax_dist[a_r, a_c])
                ax_dist[a_r, a_c].legend()
                
                #simdata_obj.plot_conddiff_dist(ax_dist_g[a_r, a_c])
                
                paradox_score[ij1, ij2, ii] = simdata_obj.paradox_score
                
                simdata_obj.plot_fr_dist(ax_base[a_r, a_c])
                
                simdata_obj.plot_box_frdiff(ax_box[:, ij2], nn_stim)
                
                #simdata_obj.plot_box_conddiff(ax_box_g[:, ij2], nn_stim)
                
                simdata_obj.plot_basefr_frdiff(ax_base_frdiff[:, ii])
                ax_base_frdiff[0, ii].set_title("pert={:.0f}%".format(nn_stim/NI*100))
                
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
                
                path_raster_fig = simdata_obj.create_fig_subdir(fig_path, "raster_dir")
                simdata_obj.plot_raster(nn_stim, ax_raster)
                fig_raster.savefig(os.path.join(path_raster_fig,
                                                "Be{}-Bi{}-P{}.png".format(Be, Bi, nn_stim)),
                                   format="png")
                plt.close(fig_raster)
                
                path_raster_fig = simdata_obj.create_fig_subdir(fig_path, "raster_dir_sep")
                simdata_obj.plot_raster_abs_chgs_all(nn_stim, ax_raster_sep)
                fig_raster_sep.savefig(os.path.join(path_raster_fig,
                                                "Be{}-Bi{}-P{}.png".format(Be, Bi, nn_stim)),
                                    format="png")
                plt.close(fig_raster_sep)
                
                ax[a_r, a_c].set_title('P={}'.format(nn_stim))
                ax_dist[a_r, a_c].set_title('P={}'.format(nn_stim))
                ax_base[a_r, a_c].set_title('P={}'.format(nn_stim))
                
            
            ax_base_frdiff[1, 2].set_xlabel("Baseline firing rate (sp/s)")
            ax_base_frdiff[1, 0].set_ylabel(r"$\Delta FR_E (sp/s)$")
            ax_base_frdiff[0, 0].set_ylabel(r"$\Delta FR_I (sp/s)$")
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
            
            fig_dist_g.savefig(os.path.join(fig_path, "g-diff-dist-Be{}-Bi{}.pdf".format(Be, Bi)),
                               format="pdf")
            
            fig_base.savefig(os.path.join(fig_path, "fr-base-dist-Be{}-Bi{}.pdf".format(Be, Bi)),
                             format="pdf")
            
            fig_avg_fr.savefig(os.path.join(fig_path, "avgfr-Be{}-Bi{}.pdf".format(Be, Bi)),
                               format="pdf")
            
            fig_base_frdiff.savefig(os.path.join(fig_path, "basevsdiff-Be{}-Bi{}.pdf".format(Be, Bi)),
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
            
        ax_box[-1, 2].set_xlabel("Number of perturbed Is")
        ax_box[0, 0].set_ylabel(r"$\Delta FR_I$")
        ax_box[1, 0].set_ylabel(r"$\Delta FR_E$")
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
    
if __name__=='__main__':

    if len(sys.argv) == 1:
        job_id = 0; num_jobs = 1
    else:
        job_id = int(sys.argv[1])
        num_jobs = int(sys.argv[2])
        
    sim_suffixes = ['sim_res_ca3_Be{:.2f}_Bi{:.2f}', 'sim_res_Be{:.2f}_Bi{:.2f}']
    
    EE_probchg_comb = EE_probchg_comb.flatten()[job_id::num_jobs]
    EI_probchg_comb = EI_probchg_comb.flatten()[job_id::num_jobs]
    II_condchg_comb = II_condchg_comb.flatten()[job_id::num_jobs]
    #fr_chg_comb = fr_chg_comb.flatten()[job_id::num_jobs]
    E_extra_comb = E_extra_comb.flatten()[job_id::num_jobs]
    bkg_chg_comb = bkg_chg_comb.flatten()[job_id::num_jobs]
    
    for ij1 in range(EE_probchg_comb.size):
        '''
        sim_suffix = "-EIeqpert-bkgfac{:.2f}-Epertfac{:.1f}-longersim-HEEcond-EE_probchg{:.2f}-EI_probchg{:.2f}".format(bkg_chg_comb[ij1],
                                                                 E_extra_comb[ij1],
                                                                 EE_probchg_comb[ij1],
                                                                 EI_probchg_comb[ij1])
        '''
        
        for sim_suf in sim_suffixes:  
            print('sim_suf={}'.format(sim_suf))        
            if 'ca3' in sim_suf:
                fig_ca = 'ca3'
            else:
                fig_ca = 'ca1'
        
            sim_suffix = "-Lbkgca3toca1-CA3eqpert-bi{:.2f}-be{:.2f}-ca1bkgfr{:.0f}-Epertfac{:.1f}-EE_probchg{:.2f}-EI_probchg{:.2f}".format(Bi_ca3, Be_ca3, r_bkg_ca1, E_extra_comb[ij1], EE_probchg_comb[ij1], EI_probchg_comb[ij1])
                     
            run_for_each_parset(sim_suffix, fig_ca)
