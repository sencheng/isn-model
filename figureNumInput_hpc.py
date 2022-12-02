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
from figureNumInput import simdata, frchg_vs_EtoI,propposfrchg, frchg, significant_proportions

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
    
    cv_e = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 3))
    cv_i = np.zeros_like(cv_e)
    
    ff_e = np.zeros_like(cv_e)
    ff_i = np.zeros_like(cv_e)
    
    pos_prop = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 2))
    mean_fr  = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 2))
    
    significant_inc = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 2))
    significant_dec = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 2))
    non_significant = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size, 2))
    
    paradox_score = np.zeros((Be_rng.size, Bi_rng.size, nn_stim_rng.size))
    
    '''
    Analysis directories
    '''
    
    diff_dists = "Distribution_Of_Differences"
    # diff_boxs  = "Box_Of_Differences"
    avg_plots  = "Average_FiringRates"
    # base_diff_rel = "Base_vs_Diff"
    # base_dist  = "Baseline_FiringRates_Dist"
    # other      = "Other_Analyses"
    
    if fig_ca == 'ca3':
        BE = np.random.choice(Be_rng, 1, replace=False)
    else:
        BE = Be_rng
    for ij1, Be in enumerate(BE):
        
        fig_box, ax_box = plt.subplots(nrows=2, ncols=Bi_rng.size,
                                        sharex=True, sharey=True,
                                        figsize=(6, 6))
        
        fig_violin, ax_violin = plt.subplots(nrows=2, ncols=Bi_rng.size,
                                              sharex=True, sharey=True,
                                              figsize=(6, 6))
        
        fig_box_mean, ax_box_mean = plt.subplots(nrows=2, ncols=Bi_rng.size,
                                                  sharex=True, sharey=True,
                                                  figsize=(6, 6))
        
        fig_violin_mean, ax_violin_mean = plt.subplots(nrows=2, ncols=Bi_rng.size,
                                                        sharex=True, sharey=True,
                                                        figsize=(6, 6))
        
        fig_box_g, ax_box_g = plt.subplots(nrows=2, ncols=Bi_rng.size,
                                            sharex=True, sharey=True)
        
        if Bi_rng.size == 1:
            ax_box = ax_box.reshape(-1, 1)
            ax_violin = ax_violin.reshape(-1, 1)
            ax_box_mean = ax_box_mean.reshape(-1, 1)
            ax_violin_mean = ax_violin_mean.reshape(-1, 1)
            ax_box_g = ax_box_g.reshape(-1, 1)
        
        for ij2, Bi in enumerate(Bi_rng):
            
            os.chdir(os.path.join(data_dir, res_dir + sim_suffix))
            sim_name = file_name.format(Be, Bi)
            print('Reading {} ...\n'.format(sim_name))
            # fl = open(sim_name, 'rb'); sim_res = pickle.load(fl); fl.close()
            
            fig, ax = plt.subplots(nrows=1, ncols=nn_stim_rng.size,
                                   sharex=True, sharey=True, figsize=(6, 3))
            fig_e, ax_e = plt.subplots(nrows=1, ncols=nn_stim_rng.size,
                                       sharex=True, sharey=True, figsize=(6, 3))
            fig_base, ax_base = plt.subplots(nrows=1, ncols=nn_stim_rng.size,
                                             sharex=True, sharey=True, figsize=(6, 3))
            fig_dist, ax_dist = plt.subplots(nrows=1, ncols=nn_stim_rng.size,
                                             sharex=True, sharey=True, figsize=(10, 4))
            fig_dist_mean, ax_dist_mean = plt.subplots(nrows=1, ncols=nn_stim_rng.size,
                                                       sharex=False, sharey=True, figsize=(10, 4))
            fig_dist_mean_pert, ax_dist_mean_pert = plt.subplots(nrows=1, ncols=nn_stim_rng.size,
                                                       sharex=False, sharey=True, figsize=(10, 4))
            fig_dist_mean_pert_line, ax_dist_mean_pert_line = plt.subplots(nrows=1, ncols=nn_stim_rng.size,
                                                       sharex=False, sharey=True)#, figsize=(10, 4))
            fig_dist_mean_sample, ax_dist_mean_sample = plt.subplots(nrows=1, ncols=nn_stim_rng.size,
                                                                     sharex=True, sharey=True, figsize=(6, 3))
            fig_dist_g, ax_dist_g = plt.subplots(nrows=1, ncols=nn_stim_rng.size,
                                                 sharex=True, sharey=True, figsize=(6, 3))
            fig_i_fr, ax_i_fr = plt.subplots(nrows=2, ncols=nn_stim_rng.size, sharex=True, sharey='row')
            fig_e_fr, ax_e_fr = plt.subplots(nrows=2, ncols=nn_stim_rng.size, sharex=True, sharey='row')
            fig_base_frdiff, ax_base_frdiff = plt.subplots(nrows=2, ncols=nn_stim_rng.size, sharex=True, sharey=True)
            
            fig_avg_fr, ax_avg_fr = plt.subplots()
            
            simdata_obj = simdata(file_name, Be, Bi)
            
            diff_dists_fig = simdata_obj.create_fig_subdir(fig_path, diff_dists)
            # diff_boxs_fig = simdata_obj.create_fig_subdir(fig_path, diff_boxs)
            # avg_plots_fig = simdata_obj.create_fig_subdir(fig_path, avg_plots)
            # base_diff_rel_fig = simdata_obj.create_fig_subdir(fig_path, base_diff_rel)
            # base_dist_fig = simdata_obj.create_fig_subdir(fig_path, base_dist)
            # other_fig = simdata_obj.create_fig_subdir(fig_path, other)
            
            ax[0].set_ylabel('E to I')
            ax[0].set_ylabel('E to I')
            
            ax_i_fr[0, 0].set_ylabel(r'$\sum (spikes_I)$ to I')
            ax_i_fr[1, 0].set_ylabel(r'$\sum (spikes_E)$ to I')
            
            ax_e_fr[0, 0].set_ylabel(r'$\sum (spikes_I)$ to E')
            ax_e_fr[1, 0].set_ylabel(r'$\sum (spikes_E)$ to E')
            
            ax_e[0].set_ylabel('E to E')
            ax_e[0].set_ylabel('E to E')
            
            ax[int(nn_stim_rng.size/2)].set_xlabel(r'$\Delta FR (sp/s)$')
            ax_e[int(nn_stim_rng.size/2)].set_xlabel(r'$\Delta FR (sp/s)$')
            ax_i_fr[1, 1].set_xlabel(r'$\Delta FR (sp/s)$')
            ax_e_fr[1, 1].set_xlabel(r'$\Delta FR (sp/s)$')
            
            ax_base[int(nn_stim_rng.size/2)].set_xlabel('Firing rate (sp/s)')
            
            ax_dist[int(nn_stim_rng.size/2)].set_xlabel(r'$\Delta FR (sp/s)$')
            
            for ii, nn_stim in enumerate(nn_stim_rng):
                
                fig_raster, ax_raster = plt.subplots(nrows=simdata_obj.Ntrials, ncols=1,
                                                     sharex=True, sharey=True)
                
                fig_raster_sep, ax_raster_sep = plt.subplots(nrows=2, ncols=2,
                                                             sharex=True)
                
                fig_raster_extremes, ax_raster_extremes = plt.subplots(nrows=2, ncols=2,
                                                             sharex=True)
                                                             
                fig_rate_extremes, ax_rate_extremes = plt.subplots(nrows=2, ncols=2,
                                                             sharex=True)
                
#                a_r, a_c = ii//3, ii%3
                
                simdata_obj.get_fr_diff(nn_stim, significance_test=significance_test)
                
                # simdata_obj.get_cond_diff(nn_stim)
                # simdata_obj.get_indegree()
                # simdata_obj.plot_indeg_frdiff(ax[a_r, a_c])
                # simdata_obj.plot_indeg_frdiff_e(ax_e[a_r, a_c])
                
                simdata_obj.plot_frdiff_dist(ax_dist[ii])
                ax_dist[ii].legend()

                simdata_obj.plot_frdiffmean_dist(ax_dist_mean[ii], nn_stim)
                ax_dist_mean[-1].legend()
                
                simdata_obj.plot_frdiffmean_dist_pertdistinct(ax_dist_mean_pert[ii], nn_stim, fig_ca)
                ax_dist_mean_pert[-1].legend(bbox_to_anchor=(1., 1),
                                             loc='upper left',
                                             borderaxespad=0.)
                
                simdata_obj.plot_frdiffmean_dist_pertdistinct_line(ax_dist_mean_pert_line[ii], nn_stim, fig_ca)
                ax_dist_mean_pert_line[-1].legend(bbox_to_anchor=(1., 1),
                                                  loc='upper left',
                                                  borderaxespad=0.)
                ax_dist_mean_pert[ii].set_ylim((0, 0.3))
                simdata_obj.plot_frdiffmean_dist_pertdistinct_line_by_model(nn_stim, fig_ca, diff_dists_fig)
                
                simdata_obj.plot_frdiffmean_samplesize_dist(ax_dist_mean_sample[ii])
                ax_dist_mean_sample[ii].legend()
                
                #simdata_obj.plot_conddiff_dist(ax_dist_g[a_r, a_c])
                
                paradox_score[ij1, ij2, ii] = simdata_obj.paradox_score
                
                simdata_obj.plot_fr_dist(ax_base[ii])
                
                simdata_obj.plot_box_frdiff(ax_box[:, ij2], nn_stim)

                simdata_obj.plot_box_frdiffmean(ax_box_mean[:, ij2], nn_stim)
                
                simdata_obj.plot_violin_frdiff(ax_violin[:, ij2], nn_stim)

                simdata_obj.plot_violin_frdiffmean(ax_violin_mean[:, ij2], nn_stim)
                
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
                
                if significance_test:
                
                    sig_exc, sig_inh = simdata_obj.get_significant_changes()
                    significant_dec[ij1, ij2, ii, 0] = sig_exc[0]/np.sum(sig_exc)
                    significant_inc[ij1, ij2, ii, 0] = sig_exc[1]/np.sum(sig_exc)
                    non_significant[ij1, ij2, ii, 0] = sig_exc[2]/np.sum(sig_exc)
                    
                    significant_dec[ij1, ij2, ii, 1] = sig_inh[0]/np.sum(sig_inh)
                    significant_inc[ij1, ij2, ii, 1] = sig_inh[1]/np.sum(sig_inh)
                    non_significant[ij1, ij2, ii, 1] = sig_inh[2]/np.sum(sig_inh)
                
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
                
                # path_raster_fig = simdata_obj.create_fig_subdir(fig_path, "raster_dir")
                # simdata_obj.plot_raster(nn_stim, ax_raster)
                # fig_raster.savefig(os.path.join(path_raster_fig,
                #                                 "Be{:.2f}-Bi{:.2f}-P{}.pdf".format(Be, Bi, nn_stim)),
                #                    format="pdf")
                # plt.close(fig_raster)
                
                # path_raster_fig = simdata_obj.create_fig_subdir(fig_path, "raster_dir_sep")
                # path_raster_fig_ext = simdata_obj.create_fig_subdir(fig_path, "raster_dir_extremes")
                # simdata_obj.plot_raster_abs_chgs_all(nn_stim, ax_raster_sep)
                # simdata_obj.plot_raster_sample_high_chgs(nn_stim, ax_raster_extremes, ax_rate_extremes)
                # fig_raster_sep.savefig(os.path.join(path_raster_fig,
                #                                 "Be{:.2f}-Bi{:.2f}-P{}.pdf".format(Be, Bi, nn_stim)),
                #                     format="pdf")
                # fig_raster_extremes.savefig(os.path.join(path_raster_fig_ext,
                #                                 "Be{:.2f}-Bi{:.2f}-P{}.pdf".format(Be, Bi, nn_stim)),
                #                     format="pdf")
                # fig_rate_extremes.savefig(os.path.join(path_raster_fig_ext,
                #                           "Avgfr-Be{:.2f}-Bi{:.2f}-P{}.pdf".format(Be, Bi, nn_stim)),
                #                           format="pdf")
                # plt.close(fig_raster_sep)
                # plt.close(fig_raster_extremes)
                # plt.close(fig_rate_extremes)
                
                ax[ii].set_title('Pert={}'.format(nn_stim))
                ax_dist[ii].set_title('Pert={}'.format(nn_stim))
                ax_dist_mean[ii].set_title('Pert={:.0f}%'.format(nn_stim/NI*100))
                ax_dist_mean_pert[ii].set_title('Pert={:.0f}%'.format(nn_stim/NI*100))
                ax_dist_mean_pert_line[ii].set_title('Pert={:.0f}%'.format(nn_stim/NI*100))
                ax_dist_mean_sample[ii].set_title('Pert={:.0f}%'.format(nn_stim/NI*100))
                ax_base[ii].set_title('P={}'.format(nn_stim))
                
                ax_dist_mean[int(nn_stim_rng.size/2)].set_xlabel(r"$\Delta FR (sp/s)$")
                ax_dist_mean[0].set_ylabel("Count")
                
                ax_dist_mean_pert[int(nn_stim_rng.size/2)].set_xlabel(r"$\Delta FR (sp/s)$")
                ax_dist_mean_pert[0].set_ylabel("Count")
                
                ax_dist_mean_pert_line[int(nn_stim_rng.size/2)].set_xlabel(r"$\Delta FR (sp/s)$")
                ax_dist_mean_pert_line[0].set_ylabel("Proportion")
                
                to_square_plots(ax_dist_mean_pert)
                boxoff(ax_dist_mean_pert)
                
                to_square_plots(ax_dist_mean_pert_line)
                boxoff(ax_dist_mean_pert_line)
            
            ax_base_frdiff[1, 1].set_xlabel("Baseline firing rate (sp/s)")
            ax_base_frdiff[1, 0].set_ylabel(r"$\Delta FR_E (sp/s)$")
            ax_base_frdiff[0, 0].set_ylabel(r"$\Delta FR_I (sp/s)$")
            
            ax_box[0, ij2].set_title('Bi={:.2f}'.format(Bi))
            ax_box[1, ij2].xaxis.set_tick_params(rotation=90)
            
            ax_box_mean[0, ij2].set_title('Bi={:.2f}'.format(Bi))
            ax_box_mean[1, ij2].xaxis.set_tick_params(rotation=90)
            
            ax_violin[0, ij2].set_title('Bi={:.2f}'.format(Bi))
            ax_violin[1, ij2].xaxis.set_tick_params(rotation=90)
            
            ax_violin_mean[0, ij2].set_title('Bi={:.2f}'.format(Bi))
            ax_violin_mean[1, ij2].xaxis.set_tick_params(rotation=90)
            
            simdata_obj.plot_avg_frs(ax_avg_fr)
            ax_avg_fr.set_title("Be={:.2f}, Bi={:.2f}".format(Be, Bi))
            ax_avg_fr.set_xlabel("Time (ms)")
            ax_avg_fr.set_ylabel("Average firing rate (sp/s)")
            
            # fig.savefig(os.path.join(other_fig, "fr-Ninp-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #             format="pdf")
            
            # fig_e.savefig(os.path.join(other_fig, "fre-Ninp-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #             format="pdf")
            
            # fig_e_fr.suptitle("Excitatory neurons")
            # fig_e_fr.savefig(os.path.join(other_fig, "fre-fr-diff-dist-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #                  format="pdf")
            
            # fig_i_fr.suptitle("Inhibitory neurons")
            # fig_i_fr.savefig(os.path.join(other_fig, "fr-fr-diff-dist-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #                  format="pdf")
            
            fig_dist.savefig(os.path.join(diff_dists_fig, "fr-diff-dist-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
                             format="pdf")

            fig_dist_mean.savefig(os.path.join(diff_dists_fig, "fr-diff-dist-mean-Be{}-Bi{}.pdf".format(Be, Bi)),
                                  format="pdf")
            
            fig_dist_mean_pert.savefig(os.path.join(diff_dists_fig, "fr-diff-dist-mean-pert-Be{}-Bi{}.pdf".format(Be, Bi)),
                                       format="pdf")
            
            fig_dist_mean_pert_line.savefig(os.path.join(diff_dists_fig, "fr-diff-dist-mean-pert-line-Be{}-Bi{}.pdf".format(Be, Bi)),
                                            format="pdf")
            
            fig_dist_mean_sample.savefig(os.path.join(diff_dists_fig, "fr-diff-dist-mean-samples-Be{}-Bi{}.pdf".format(Be, Bi)),
                                         format="pdf")
            
            # fig_dist_g.savefig(os.path.join(other_fig, "g-diff-dist-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #                    format="pdf")
            
            # fig_base.savefig(os.path.join(base_dist_fig, "fr-base-dist-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #                  format="pdf")
            
            # fig_avg_fr.savefig(os.path.join(avg_plots_fig, "avgfr-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #                    format="pdf")
            # -Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #             format="pdf")
            
            # fig_e.savefig(os.path.join(other_fig, "fre-Ninp-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #             format="pdf")
            
            # fig_e_fr.suptitle("Excitatory neurons")
            # fig_e_fr.savefig(os.path.join(other_fig, "fre-fr-diff-dist-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #                  format="pdf")
            
            # fig_i_fr.suptitle("Inhibitory neurons")
            # fig_i_fr.savefig(os.path.join(other_fig, "fr-fr-diff-dist-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #                  format="pdf")
            # fig_base_frdiff.savefig(os.path.join(base_diff_rel_fig, "basevsdiff-Be{:.2f}-Bi{:.2f}.pdf".format(Be, Bi)),
            #                    format="pdf")
                    
            plt.close(fig)
            plt.close(fig_e)
            plt.close(fig_dist)
            plt.close(fig_dist_g)
            plt.close(fig_base)
            plt.close(fig_i_fr)
            plt.close(fig_e_fr)
            plt.close(fig_avg_fr)
            plt.close(fig_dist_mean)
            plt.close(fig_dist_mean_pert)
            plt.close(fig_dist_mean_sample)
            plt.close(fig_base_frdiff)
            
        # ax_box[-1, int(Bi_rng.size/2)].set_xlabel("Percent of CA3 neurons perturbed")
        # ax_box[0, 0].set_ylabel(r"$\Delta FR_I$")
        # ax_box[1, 0].set_ylabel(r"$\Delta FR_E$")
        # fig_box.suptitle("Be={:.2f}".format(Be))
            
        # fig_box.savefig(os.path.join(diff_boxs_fig, "fr-diff-box-Be{:.2f}.pdf".format(Be)),
        #                  format="pdf")
        
        # ax_violin[-1, int(Bi_rng.size/2)].set_xlabel("Percent of CA3 neurons perturbed")
        # ax_violin[0, 0].set_ylabel(r"$\Delta FR_I$")
        # ax_violin[1, 0].set_ylabel(r"$\Delta FR_E$")
        # fig_violin.suptitle("Be={:.2f}".format(Be))
            
        # fig_violin.savefig(os.path.join(diff_boxs_fig, "fr-diff-violin-Be{:.2f}.pdf".format(Be)),
        #                  format="pdf")
        
        # ax_box_mean[-1, int(Bi_rng.size/2)].set_xlabel("Percent of CA3 neurons perturbed")
        # ax_box_mean[0, 0].set_ylabel(r"$\Delta FR_I$")
        # ax_box_mean[1, 0].set_ylabel(r"$\Delta FR_E$")
        # fig_box_mean.suptitle("Be={:.2f}".format(Be))
            
        # fig_box_mean.savefig(os.path.join(diff_boxs_fig, "fr-diffmean-box-Be{:.2f}.pdf".format(Be)),
        #                  format="pdf")
        
        # ax_violin_mean[-1, int(Bi_rng.size/2)].set_xlabel("Percent of CA3 neurons perturbed")
        # ax_violin_mean[0, 0].set_ylabel(r"$\Delta FR_I$")
        # ax_violin_mean[1, 0].set_ylabel(r"$\Delta FR_E$")
        # fig_violin_mean.suptitle("Be={:.2f}".format(Be))
            
        # fig_violin_mean.savefig(os.path.join(diff_boxs_fig, "fr-diffmean-violin-Be{:.2f}.pdf".format(Be)),
        #                  format="pdf")
        
        fig_box_g.savefig(os.path.join(fig_path, "g-diff-box-Be{:.2f}.pdf".format(Be)),
                          format="pdf")
        
        plt.close(fig_box)
        plt.close(fig_box_mean)
        plt.close(fig_box_g)
        plt.close(fig_violin_mean)
        plt.close(fig_violin)
        
        
    frchgdata = {'E': {'proportion_increase': pos_prop[:,:,:,0],
                       'mean_change': mean_fr[:,:,:,0]},
                 'I': {'proportion_increase': pos_prop[:,:,:,1],
                       'mean_change': mean_fr[:,:,:,1]}}
    
    fig_frchg_ei_e = frchg_vs_EtoI(frchgdata)
    # fig_frchg_ei_e.savefig(os.path.join(other_fig, "frchg-EtoI-E.pdf"),
    #                        format="pdf")
    
    # fig_frchg_ei_e = frchg_vs_EtoI(frchgdata, "I")
    # fig_frchg_ei_e.savefig(os.path.join(other_fig, "frchg-EtoI-I.pdf"),
    #                        format="pdf")

    # fig_posprop = propposfrchg(frchgdata)
    # fig_posprop.savefig(os.path.join(other_fig, "propposfrchg.pdf"),
    #                     format="pdf")
    
    # fig_frchg = frchg(frchgdata)
    # fig_frchg.savefig(os.path.join(other_fig, "frchg.pdf"),
    #                     format="pdf")
    
    # fig_sig_exc = significant_proportions([significant_dec[:, :, :, 0],
    #                                        significant_inc[:, :, :, 0],
    #                                        non_significant[:, :, :, 0]])
    
    # fig_sig_exc.savefig(os.path.join(other_fig, "change-fr-prop-exc.pdf"),
    #                     format="pdf")
    
    # fig_sig_inh = significant_proportions([significant_dec[:, :, :, 1],
    #                                        significant_inc[:, :, :, 1],
    #                                        non_significant[:, :, :, 1]])
    
    # fig_sig_inh.savefig(os.path.join(other_fig, "change-fr-prop-inh.pdf"),
    #                     format="pdf")
    
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
    
    # Ge = np.append(0, Be_rng)
    # Gi = np.append(0, Bi_rng)
    
    # fig_psc, ax_psc = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    # bar_psc = fig_psc.add_axes([0.7, 0.15, 0.02, 0.3])
        
    # for ii, nn_stim in enumerate(nn_stim_rng):
        
    #     # if paradox_score[:, :, ii].max()>1:
    #     f = ax_psc[ii//3, ii%3].pcolormesh(Gi, Ge, paradox_score[:, :, ii],
    #                                        vmin=paradox_score.min(),
    #                                        vmax=paradox_score.max())
    #     ax_psc[ii//3, ii%3].set_title("pert={:.0f}%".format(nn_stim/NI*100))
        
    # ax_psc[1, 0].set_xlabel("Inh. Cond.")
    # ax_psc[1, 1].set_xlabel("Inh. Cond.")
    
    # ax_psc[0, 0].set_ylabel("Exc. Cond.")
    # ax_psc[1, 0].set_ylabel("Exc. Cond.")
    
    # if "f" in locals():
    #     plt.colorbar(f, cax=bar_psc)
    
    # fig_psc.savefig(os.path.join(other_fig, "paradoxical-score.pdf"), format="pdf")
        
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
    with open("simres_obj.pkl", "wb") as obj_fl:
        pickle.dump(simdata_obj, obj_fl, pickle.HIGHEST_PROTOCOL)
    os.chdir(cwd)
    
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
