import numpy as np
import pickle
import copy
import os
from imp import reload
import defaultParams; reload(defaultParams); from defaultParams import *
import searchParams; reload(searchParams); from searchParams import *

def _read_data(reg, be, bi, res_path):
    sim_res = {}
    for rng_c in C_rng:
        if reg == 'ca3':
            sim_name = 'sim_res_{}_Be{:.2f}_Bi{:.2f}_Mo{:d}'.format(reg, be, bi, rng_c)
        else:
            sim_name = 'sim_res_Be{:.2f}_Bi{:.2f}_Mo{:d}'.format(be, bi, rng_c)
        if os.path.exists(os.path.join(res_path, sim_name)):
            print('\nLoading {} ...'.format(sim_name))
        else:
            print("The file does not exist, terminating ...")
        with open(os.path.join(res_path, sim_name), 'rb') as fl:
            tmp_out = pickle.load(fl)
        if rng_c == C_rng[0]:
            sim_res = copy.copy(tmp_out)
        for nn_stim in nn_stim_rng:
            for tr in tmp_out[nn_stim][2].keys():
                sim_res[nn_stim][2][tr] = copy.copy(tmp_out[nn_stim][2][tr])
    return sim_res

def _remove_data(reg, res_path):
    for be in Be_rng:
        for bi in Bi_rng:
            for rng_c in C_rng:
                if reg == 'ca3':
                    sim_name = 'sim_res_{}_Be{:.2f}_Bi{:.2f}_Mo{:d}'.format(reg, be, bi, rng_c)
                else:
                    sim_name = 'sim_res_Be{:.2f}_Bi{:.2f}_Mo{:d}'.format(be, bi, rng_c)
                if os.path.exists(os.path.join(res_path, sim_name)):
                    print('\nDeleting {} ...'.format(sim_name))                    
                    os.remove(os.path.join(res_path, sim_name))
                else:
                  print("The file does not exist")
                
Be_rng_comb, Bi_rng_comb, EE_probchg_comb, EI_probchg_comb, II_condchg_comb, E_extra_comb, bkg_chg_comb, C_rng_comb = np.meshgrid(Be_rng, Bi_rng, EEconn_chg_factor, EIconn_chg_factor, IIconn_chg_factor, E_extra_stim_factor, bkg_chg_factor, C_rng)
# Be_rng_comb, Bi_rng_comb, EE_probchg_comb, EI_probchg_comb, II_condchg_comb, E_extra_comb, bkg_chg_comb, CA3_CP_comb = np.meshgrid(Be_rng, Bi_rng, EEconn_chg_factor, EIconn_chg_factor, IIconn_chg_factor, E_extra_stim_factor, bkg_chg_factor, CA3_conn_prob_fac)

Be_rng_comb = Be_rng_comb.flatten()
Bi_rng_comb = Bi_rng_comb.flatten()
EE_probchg_comb = EE_probchg_comb.flatten()
EI_probchg_comb = EI_probchg_comb.flatten()
II_condchg_comb = II_condchg_comb.flatten()
E_extra_comb = E_extra_comb.flatten()
bkg_chg_comb = bkg_chg_comb.flatten()
C_rng_comb = C_rng_comb.flatten()
# CA3_CP_comb = CA3_CP_comb.flatten()

for reg in ['ca1', 'ca3']:
    for ij1 in range(EE_probchg_comb.size):
        sim_suffix_comp = sim_suffix.format(extra_bkg_e, E3E1_cond_chg, Bi_ca3*EI_probchg_comb[ij1], Be_ca3*EE_probchg_comb[ij1], r_bkg_ca1, E_extra_comb[ij1], EE_probchg_comb[ij1], EI_probchg_comb[ij1])
        res_path = os.path.join(data_dir, res_dir+sim_suffix_comp)
        print('Processing region: {} and processing directory {}'.format(reg, res_path))
        if not os.path.exists(res_path): print('Simulation data does not exist!')
        for be in Be_rng:
            for bi in Bi_rng:
                data = _read_data(reg, be, bi, res_path)
                if reg == 'ca1':
                    sim_name = 'sim_res_Be{:.2f}_Bi{:.2f}'.format(be, bi)
                else:
                    sim_name = 'sim_res_{}_Be{:.2f}_Bi{:.2f}'.format(reg, be, bi)
                print('\nWriting {}'.format(sim_name))
                with open(os.path.join(res_path, sim_name), 'wb') as fl:
                    pickle.dump(data, fl)
        #_remove_data(reg, res_path)
