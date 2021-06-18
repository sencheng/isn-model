################################################################################
# -- Simulating Exc-Inh spiking networks in response to inhibitory perturbation
################################################################################

import numpy as np; import pylab as pl; import time, os, sys, pickle
from scipy.stats import norm
from imp import reload
import defaultParams_3INTtype; reload(defaultParams_3INTtype); from defaultParams_3INTtype import *;
import networkTools; reload(networkTools); import networkTools as net_tools
import nest

cwd = os.getcwd()

t_init = time.time()

################################################################################
#### functions

# -- rectification
def _rect_(xx): return xx*(xx>0)

# -- generates the weight matrix
def _mycon_(N1, N2, B12, B12_std, pr=1.):
    zb = np.random.binomial(1, pr, (N1,N2))
    zw = np.sign(B12) * _rect_(np.random.normal(abs(B12),abs(B12_std),(N1,N2)))
    zz = zb* zw
    return zz

def _guasconn_(N1, N2, B12, conn_std, pr=1.):
    
    '''
    This function is very similar the "_mycon_". The reason I added this function
    is that the sparse connections in "_mycon_" are created using a binomial
    process. Changing the variance in binomial process requires changing "n, p",
    which consequently changes the total number of input connections and thus
    lead to a different network behavior. To overcome this limitation, I used 
    a Guassian process instead of the binomial one, which enables me to change
    the variance while keeping the total number of incoming connections almost
    fixed.
    '''
    
    zb = np.zeros((N1, N2))
    num_in = np.random.normal(N1*pr, conn_std, N2)
    num_in[num_in<0] = 0
    num_in_i = num_in.astype(int)
    
    for n2 in range(N2):
        rp = np.random.permutation(N1)[:num_in_i[n2]]
        zb[rp, n2] = 1
    
    zw = np.sign(B12) * _rect_(np.random.normal(abs(B12),abs(B12/5),(N1,N2)))
    zz = zb * zw
    return zz

# -- runs a network simulation with a defined inh perturbation
bw = 50.
def myRun(rr1, rr2, Tstim=Tstim, Tblank=Tblank, Ntrials=Ntrials, bw = bw, \
            rec_conn={'EtoE':1, 'EtoI':1, 'ItoE':1, 'ItoI':1}, nn_stim=0):

        # -- restart the simulator
        net_tools._nest_start_()
        init_seed = np.random.randint(1, 1234, n_cores)
        print('init_seed = ', init_seed)
        nest.SetStatus([0],[{'rng_seeds':init_seed.tolist()}])

        # -- exc & inh neurons
        exc_neurons = net_tools._make_neurons_(NE, neuron_model=cell_type, \
                                               myparams={'b':NE*[0.], 'a':NE*[0.]})
        inh_neurons_pv = net_tools._make_neurons_(NI_pv, neuron_model=cell_type, \
                                                  myparams={'b':NE*[0.],'a':NE*[0.]})
        inh_neurons_som = net_tools._make_neurons_(NI_som, neuron_model=cell_type, \
                                                   myparams={'b':NE*[0.],'a':NE*[0.]})
        inh_neurons_vip = net_tools._make_neurons_(NI_vip, neuron_model=cell_type, \
                                                   myparams={'b':NE*[0.],'a':NE*[0.]})

        all_neurons = exc_neurons + inh_neurons_pv + inh_neurons_som + inh_neurons_vip

        # -- recurrent connectivity
        # if rec_conn['EtoE']:
        #     net_tools._connect_pops_(exc_neurons, exc_neurons, W_EtoE)
        # if rec_conn['EtoI']:
        #     net_tools._connect_pops_(exc_neurons, inh_neurons, W_EtoI)
        # if rec_conn['ItoE']:
        #     net_tools._connect_pops_(inh_neurons, exc_neurons, W_ItoE)
        # if rec_conn['ItoI']:
        #     net_tools._connect_pops_(inh_neurons, inh_neurons, W_ItoI)
        
        net_tools._connect_pops_(exc_neurons, exc_neurons, W_EtoE)
        
        net_tools._connect_pops_(exc_neurons, inh_neurons_pv, W_EtoIpv)
        net_tools._connect_pops_(exc_neurons, inh_neurons_som, W_EtoIsom)
        net_tools._connect_pops_(exc_neurons, inh_neurons_vip, W_EtoIvip)
        
        net_tools._connect_pops_(inh_neurons_pv, exc_neurons, W_IpvtoE)
        net_tools._connect_pops_(inh_neurons_som, exc_neurons, W_IsomtoE)
        
        net_tools._connect_pops_(inh_neurons_pv, inh_neurons_pv, W_IpvtoIpv)
        net_tools._connect_pops_(inh_neurons_som, inh_neurons_pv, W_IsomtoIpv)
        net_tools._connect_pops_(inh_neurons_som, inh_neurons_vip, W_IsomtoIvip)
        net_tools._connect_pops_(inh_neurons_vip, inh_neurons_som, W_IviptoIsom)
        
        '''Added connections for reproduction of the original results'''
        
        net_tools._connect_pops_(inh_neurons_vip, exc_neurons, W_IviptoE)
        net_tools._connect_pops_(inh_neurons_som, inh_neurons_som, W_IsomtoIsom)
        net_tools._connect_pops_(inh_neurons_vip, inh_neurons_vip, W_IviptoIvip)
        net_tools._connect_pops_(inh_neurons_vip, inh_neurons_pv, W_IviptoIpv)
        net_tools._connect_pops_(inh_neurons_pv, inh_neurons_som, W_IpvtoIsom)
        net_tools._connect_pops_(inh_neurons_pv, inh_neurons_vip, W_IpvtoIvip)

        # -- recording spike data
        spikes_all = net_tools._recording_spikes_(neurons=all_neurons)
        
        # -- recording inhibitory current data
        currents_all = net_tools._recording_gin_(neurons=all_neurons)

        # -- background input
        pos_inp = nest.Create("poisson_generator", N)

        for ii in range(N):
            nest.Connect([pos_inp[ii]], [all_neurons[ii]], \
            syn_spec = {'weight':Be_bkg, 'delay':delay_default})

        # -- simulating network for N-trials
        for tri in range(Ntrials):
            print('')
            print('# -> trial # ', tri+1)

            ## transient
            for ii in range(N):
                nest.SetStatus([pos_inp[ii]], {'rate':rr1[ii]})
            net_tools._run_simulation_(Ttrans)

            ## baseline
            for ii in range(N):
                nest.SetStatus([pos_inp[ii]], {'rate':rr1[ii]})
            net_tools._run_simulation_(Tblank)

            ## perturbing a subset of inh
            for ii in range(N):
                nest.SetStatus([pos_inp[ii]], {'rate':rr2[ii]})
            net_tools._run_simulation_(Tstim)

        # -- reading out spiking activity
        spd = net_tools._reading_spikes_(spikes_all)
        
        # -- reading out currents
        curr = net_tools._reading_currents_(currents_all)

        # -- computes the rates out of spike data in a given time interval
        def _rate_interval_(spikedata, T1, T2, bw=bw):
            tids = (spikedata['times']>T1) * (spikedata['times']<T2)
            rr = np.histogram2d(spikedata['times'][tids], spikedata['senders'][tids], \
                 range=((T1,T2),(1,N)), bins=(int((T2-T1)/bw),N))[0] / (bw/1e3)
            return rr

        rout_blank = np.zeros((Ntrials, int(Tblank / bw), N))
        rout_stim = np.zeros((Ntrials, int(Tstim / bw), N))
        for tri in range(Ntrials):
            Tblock = Tstim+Tblank+Ttrans
            rblk = _rate_interval_(spd, Tblock*tri+Ttrans, Tblock*tri+Ttrans+Tblank)
            rstm = _rate_interval_(spd, Tblock*tri+Ttrans+Tblank, Tblock*(tri+1))
            rout_blank[tri,:,:] = rblk
            rout_stim[tri,:,:] = rstm

        print('##########')
        print('## Mean firing rates {Exc | Inh (pert.) | Inh (non-pert.)}')
        print('## Before pert.: ', \
        np.round(rout_blank[:,:,0:NE].mean(),1), \
        np.round(rout_blank[:,:,NE:NE+nn_stim].mean(),1), \
        np.round(rout_blank[:,:,NE+nn_stim:].mean(),1) )
        print('## After pert.: ', \
        np.round(rout_stim[:,:,0:NE].mean(),1), \
        np.round(rout_stim[:,:,NE:NE+nn_stim].mean(),1), \
        np.round(rout_stim[:,:,NE+nn_stim:].mean(),1) )
        print('##########')

        return rout_blank, rout_stim, spd, curr

################################################################################

if len(sys.argv) == 1:
    job_id = 0; num_jobs = 1
else:
    job_id = int(sys.argv[1])
    num_jobs = int(sys.argv[2])
    
# simulate(job_id, num_jobs)

os.chdir(cwd)

# def simulate(job_id, num_jobs):

#pert_fr = np.arange(-400, -2100, -400)
fr_chg_factor = np.arange(0.5, 1, .1)
E_extra_stim_factor = np.arange(1.2, 2.1, 0.2)
EEconn_chg_factor = np.arange(0.9, 0.95, 0.05)
EIconn_chg_factor = np.arange(2.0, 2.01, 0.1)

Be_rng_comb, Bi_rng_comb = np.meshgrid(Be_rng, Bi_rng)
Be_rng_comb = Be_rng_comb.flatten()[job_id::num_jobs]
Bi_rng_comb = Bi_rng_comb.flatten()[job_id::num_jobs]
# EE_probchg_comb = EE_probchg_comb.flatten()[job_id::num_jobs]
# EI_probchg_comb = EI_probchg_comb.flatten()[job_id::num_jobs]
#fr_chg_comb = fr_chg_comb.flatten()[job_id::num_jobs]
#E_extra_comb = E_extra_comb.flatten()[job_id::num_jobs]
#pert_comb = pert_comb.flatten()[job_id::num_jobs]

for ij1 in range(Be_rng_comb.size):
    
    #r_stim = pert_comb[ij1]
    Be, Bi = Be_rng_comb[ij1], Bi_rng_comb[ij1]
    Bee, Bei = Be, Be
    Bie, Bii = Bi, Bi

    #sim_suffix = "-pert{}".format(r_stim)
    #sim_suffix = "-EIincfac{:.3f}".format(fr_chg_comb[ij1])
    #sim_suffix = "-Iincfac{:.3f}-Ered{:.1f}".format(fr_chg_comb[ij1], E_extra_comb[ij1])
    # sim_suffix = "-EEstdfac2-HEEcond-EE_probchg{:.2f}-EI_probchg{:.2f}".format(EE_probchg_comb[ij1], EI_probchg_comb[ij1])

    print('####################')
    print('### (Be, Bi): ', Be, Bi)
    print('####################')

    # -- result path
    res_path = os.path.join(cwd, res_dir+sim_suffix)
    if not os.path.exists(res_path): os.makedirs(res_path, exist_ok=True)

    os.chdir(res_path)
    print('Resetting random seed ...')
    np.random.seed(1)
    '''
    # -- L23 recurrent connectivity
    p_conn = 0.15
    W_EtoE = _mycon_(NE, NE, Bee, Bee/5, p_conn*EE_probchg_comb[ij1])
    W_EtoI = _mycon_(NE, NI, Bei, Bei/5, p_conn*EI_probchg_comb[ij1])
    W_ItoE = _mycon_(NI, NE, Bie, Bie/5, 1.)
    W_ItoI = _mycon_(NI, NI, Bii, Bii/5, 1.)
    '''
    # Indegree with Guassian distribution
    p_conn = 0.15
    p_conn_ee = p_conn#*EE_probchg_comb[ij1]
    p_conn_ei = p_conn#*EI_probchg_comb[ij1]
    W_EtoE = _guasconn_(NE, NE, Bee, np.sqrt(NE*p_conn_ee*(1-p_conn_ee))*3, p_conn_ee)
    
    W_EtoIpv = _guasconn_(NE, NI_pv, Bei, np.sqrt(NI*p_conn_ei*(1-p_conn_ei)), p_conn_ei)
    W_EtoIsom = _guasconn_(NE, NI_som, Bei, np.sqrt(NI*p_conn_ei*(1-p_conn_ei)), p_conn_ei)
    W_EtoIvip = _guasconn_(NE, NI_vip, Bei, np.sqrt(NI*p_conn_ei*(1-p_conn_ei)), p_conn_ei)
    
    W_IpvtoE = _mycon_(NI_pv, NE, Bie, Bie/5, 1.)
    W_IsomtoE = _mycon_(NI_som, NE, Bie, Bie/5, 1.)
    
    W_IpvtoIpv = _mycon_(NI_pv, NI_pv, Bii, Bii/5, 1.)
    W_IsomtoIpv = _mycon_(NI_som, NI_pv, Bii, Bii/5, 1.)
    W_IsomtoIvip = _mycon_(NI_som, NI_vip, Bii, Bii/5, 1.)
    W_IviptoIsom = _mycon_(NI_vip, NI_som, Bii, Bii/5, 1.)
    
    
    '''Added connections for reproduction of the original results'''
    W_IviptoE = _mycon_(NI_vip, NE, Bie, Bie/5, 1.)
    W_IsomtoIsom = _mycon_(NI_som, NI_som, Bii, Bii/5, 1.)
    W_IviptoIvip = _mycon_(NI_vip, NI_vip, Bii, Bii/5, 1.)
    W_IviptoIpv = _mycon_(NI_vip, NI_pv, Bii, Bii/5, 1.)
    W_IpvtoIsom = _mycon_(NI_pv, NI_som, Bii, Bii/5, 1.)
    W_IpvtoIvip = _mycon_(NI_pv, NI_vip, Bii, Bii/5, 1.)
    # -- running simulations
    sim_res = {}

    for nn_stim in nn_stim_rng:

        print('\n # -----> size of pert. inh: ', nn_stim)

        np.random.seed(2)
        r_extra = np.zeros(N)
        r_extra[N-NI_vip:N-NI_vip+int(NI_vip*nn_stim/NI)] = r_stim
        #r_extra[NE:NE+nn_stim] = r_stim
        r_extra[0:int(NE*nn_stim/NI)] = r_stim

        #fr_inc_factor = fr_chg_comb[ij1]
        r_bkg_e = r_bkg; r_bkg_i = r_bkg
        rr1 = np.hstack((r_bkg_e*np.ones(NE), r_bkg_i*np.ones(NI)))
        #rr1 = r_bkg*np.ones(N)
        rr2 = rr1 + r_extra

        sim_res[nn_stim] = myRun(rr1, rr2, nn_stim=nn_stim)

    sim_res['nn_stim_rng'], sim_res['Ntrials'] = nn_stim_rng, Ntrials
    sim_res['N'], sim_res['NE'], sim_res['NI'] = N, NE, NI
    sim_res['NI_pv'], sim_res['Ni_som'], sim_res['NI_vip'] = NI_pv, NI_som, NI_vip
    sim_res['Tblank'], sim_res['Tstim'], sim_res['Ttrans'] = Tblank, Tstim, Ttrans
    sim_res['W_EtoE'] = W_EtoE
    
    sim_res['W_EtoIpv'], sim_res['W_EtoIsom'], sim_res['W_EtoIvip'] = W_EtoIpv, W_EtoIsom, W_EtoIvip
    
    sim_res['W_IpvtoE'], sim_res['W_IsomtoE'] = W_IpvtoE, W_IsomtoE
    
    sim_res['W_IpvtoIpv'], sim_res['W_IsomtoIpv'] = W_IpvtoIpv, W_IsomtoIpv
    sim_res['W_IsomtoIvip'], sim_res['W_IviptoIsom'] = W_IsomtoIvip, W_IviptoIsom

    os.chdir(res_path);
    sim_name = 'sim_res_Be'+str(Be)+'_Bi'+str(Bi)
    fl = open(sim_name, 'wb'); pickle.dump(sim_res, fl); fl.close()

t_end = time.time()
print('took: ', np.round((t_end-t_init)/60), ' mins')

################################################################################
################################################################################
################################################################################
