################################################################################
# -- Default set of parameters
################################################################################

import numpy as np; import pylab as pl; import time, sys, os
import matplotlib
from defaultParams import *

fr_chg_factor = np.array([1])#np.arange(0.5, 1, .1)
E_extra_stim_factor = np.array([1.0])#np.arange(0.2, 0.3, 0.2)#np.array([0.8])
EEconn_chg_factor = np.array([0.9])# np.arange(1.2, 2.1, 0.2)#np.array([0.9])
EIconn_chg_factor = np.array([3.0])#np.arange(0.6, 3, 0.2)#np.array([2.0])
IIconn_chg_factor = np.arange(1, 1.1, 0.2)
bkg_chg_factor    = np.array([1.])#np.arange(1.05, 1.11, 0.05)
E3E1_cond_chg     = 0.2

EE_probchg_comb, EI_probchg_comb, II_condchg_comb, E_extra_comb, bkg_chg_comb = \
    np.meshgrid(EEconn_chg_factor, EIconn_chg_factor, IIconn_chg_factor, E_extra_stim_factor, bkg_chg_factor)
extra_bkg_e = 2500.0
#pert_comb = pert_comb.flatten()[job_id::num_jobs]

E_pert_frac = 1.0

print("Total number of parameter combinations covered = {}".format(EE_probchg_comb.size))
