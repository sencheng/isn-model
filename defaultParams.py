################################################################################
# -- Default set of parameters
################################################################################

import numpy as np; import time, sys, os
import matplotlib.pylab as pl
import matplotlib

## the number of cores to be used for simulations
n_cores = 4

# define the NEST path if it's needed
nest_path = '/Users/sadra/NEST/nest/ins/lib/python3.4/site-packages/'
if os.path.exists(nest_path):
    sys.path.append(nest_path)
    
# Result directory
res_dir = "SimulationFiles"
fig_initial = "Figures"
sim_suffix = "-ISCA3ISN-E3extrabkg{:.0f}-E3E1fac{:.1f}-bi{:.2f}-be{:.2f}-ca1bkgfr{:.0f}-Epertfac{:.1f}-EE_probchg{:.2f}-EI_probchg{:.2f}"
data_dir = "./"#"/local2/mohammad/data/5HT2A"
fig_dir  = data_dir

#------------- neuron params

# resting potential (mV)
Ur = -70.e-3
# reversal potential of exc. (mV)
Ue = 0.e-3
# reversal potential of inh. (mV)
Ui = -75.e-3
# threshold voltage (mV)
Uth = -50.e-3
# reset potential (mV)
Ureset = -60.e-3

# membrane capacitance (F)
C = 120e-12
# leak conductance (S)
Gl = 1./140e6
# sample Exc and Inh conductances (nS)
Be, Bi = .1, -.2

# range of Exc and Inh conductances (nS)
#Be_rng = np.array([0.01, .05, .1, .15, .2, .25])
#Be_rng = np.arange(0.1, .81, 0.1)
Be_rng = np.array([0.55])
Bi_rng = np.array([-0.3])

Be_ca3 = 0.03
Bi_ca3 = -0.3
EE3_prob_chg_factor, EI3_prob_chg_factor = 2.6, 1.0

# background and stimulus conductances (nS)
Be_bkg = .1
Be_stim = .1

# exc. synaptic time constant (s)
tau_e = 1.e-3
# inh. synaptic time constant (s)
tau_i = 1.e-3

# refractory time (s)
t_ref = 2e-3

# conductance-based alpha-synapses neuron model
neuron_params_default = \
{'C_m': C*1e12,
  'E_L': Ur*1000.,
  'E_ex': Ue*1000.,
  'E_in': Ui*1000.,
  'I_e': 0.0,
  'V_m': Ur*1000.,
  'V_reset': Ureset*1000.,
  'V_th': Uth*1000.,
  'g_L': Gl*1e9,
  't_ref': t_ref*1000.,
  'tau_syn_ex': tau_e*1000.,
  'tau_syn_in': tau_i*1000.}

# -- simulation params

#default synaptic delay (ms)
delay_default = .1

# time resolution of simulations (ms)
dt = .1

# background rate (sp/s)
r_bkg = 10000.-400.
r_bkg_ca1 = 7000
# rate of perturbation (sp/s)
r_stim = -400.

# transitent time to discard the data (ms)
Ttrans = 1000
# simulation time before perturbation (ms)
Tblank= 1000.
# simulation time of and after perturbation (ms)
Tstim = 1000.

# number of trials
Ntrials = 10
#rng_conn = np.arange(1, 10.1).astype(int)
# -- network params

# fraction of Inh neurons
frac = .2
# total population size (Exc + Inh)
N = 2000
# size of Inh population
NI = int(frac*N)
# size of Exc population
NE = N - NI

# range of the size of Inh perturbations
nn_stim_rng = (np.array([0.1, .25, 0.5, .75, 1])*NI).astype('int')
#nn_stim_rng = (np.array([0.05, 0.1, 0.15, 0.2, .25])*NI).astype('int')
# single cell type
cell_type = 'aeif_cond_alpha'

# record from conductances?
rec_from_cond = False

# perturbation type
het_pert = False

# perform significance test on 
significance_test = False

SIZE = 12
pl.rc('font', size=SIZE)  # controls default text sizes
pl.rc('axes', titlesize=SIZE)  # fontsize of the axes title
pl.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
pl.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
pl.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
pl.rc('legend', fontsize=SIZE)  # legend fontsize
pl.rc('figure', titlesize=SIZE)  # fontsize of the figure title

# half-frame axes
def HalfFrame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

################################################################################
################################################################################
################################################################################
