import time
import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

import root_finding

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. a
    par.a_grid[:] = equilogspace(0.0,ss.w*par.a_max,par.Na)
    
    # b. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)
    
    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    assert np.isclose(np.sum(par.probfix),1), 'Likelihood of being in a fixed state should sum to one'

    for i_fix in range(par.Nfix):
        ss.z_trans[i_fix,:,:] = z_trans
        ss.Dz[i_fix,:] = z_ergodic
        ss.Dbeg[i_fix,:,0] = ss.Dz[i_fix,:] * par.probfix[i_fix]  # ergodic at a_lag = 0.0
        ss.Dbeg[i_fix,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # a. raw value
    y = ss.w*par.z_grid
    c = m = (1+ss.r*(1-ss.taua))*par.a_grid[np.newaxis,:] + (1-ss.taul)* y[:,np.newaxis]
    v_a = (1+ss.r*(1-ss.taua))*c**(-par.sigma)

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans@v_a

def obj_ss(K_ss,model,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    # a. production
    ss.Gamma = par.Gamma_ss # model user choice
    ss.K = K_ss
    ss.L = 1.0 # by assumption
    ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)    

    # b. implied prices
    ss.rk = par.alpha*ss.Gamma*(ss.K/ss.L)**(par.alpha-1.0)
    ss.r = ss.rk - par.delta
    ss.w = (1.0-par.alpha)*ss.Gamma*(ss.K/ss.L)**par.alpha

    # c. household behavior
    if do_print:

        print(f'guess {ss.K = :.4f}')    
        print(f'implied {ss.r = :.4f}')
        print(f'implied {ss.w = :.4f}')

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    ss.A_hh = np.sum(ss.a*ss.D) # hint: is actually computed automatically
    ss.C_hh = np.sum(ss.c*ss.D)

    if do_print: print(f'implied {ss.A_hh = :.4f}')

    # d. bonds market and taxe income

    ss.taxa =  ss.taua*ss.r*ss.A_hh
    ss.taxl =   ss.taul*ss.w* np.sum( par.zeta@(par.z_grid@ss.D) )
    ss.B = -1/ss.r * ( ss.G -ss.taxa -ss.taxl)

    if do_print: print(f'implied {ss.B = :.4f}')

    # e. market clearing
    ss.clearing_A = ss.A_hh-ss.K-ss.B

    ss.I = ss.K - (1-par.delta)*ss.K
    ss.IB = ss.r*ss.B+ss.G-ss.taxa -ss.taxl
    ss.C = ss.Y - ss.I - ss.IB - ss.G
    ss.clearing_C = ss.C_hh-ss.C

    return ss.clearing_A # target to hit
    
def find_ss(model,method='direct',do_print=False,K_min=0.1,K_max=20.0,NK=10):
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    if method == 'direct':
        find_ss_direct(model,do_print=do_print,K_min=K_min,K_max=K_max,NK=NK)
    else:
        raise NotImplementedError

    if do_print: print(f'found steady state in {elapsed(t0)}')

def find_ss_direct(model,do_print=False,K_min=1.,K_max=1.0,NK=10):
    """ find steady state using direct method """

    # a. broad search
    if do_print: print(f'### step 1: broad search ###\n')

    K_ss_vec = np.linspace(K_min,K_max,NK) # trial values
    clearing_A = np.zeros(K_ss_vec.size) # asset market errors

    for i,K_ss in enumerate(K_ss_vec):
        
        try:
            clearing_A[i] = obj_ss(K_ss,model,do_print=do_print)
        except Exception as e:
            clearing_A[i] = np.nan
            print(f'{e}')
            
        if do_print: print(f'clearing_A = {clearing_A[i]:12.8f}\n')
            
    # b. determine search bracket
    if do_print: print(f'### step 2: determine search bracket ###\n')


    #K_max, K_min = find_K_interval(K_ss_vec,clearing_A,model,do_print)
    K_max = np.min(K_ss_vec[clearing_A < 0])
    K_min = np.max(K_ss_vec[clearing_A > 0])

    if do_print: print(f'K in [{K_min:12.8f},{K_max:12.8f}]\n')

    # c. search
    if do_print: print(f'### step 3: search ###\n')

    root_finding.brentq(
        obj_ss,K_min,K_max,args=(model,),do_print=do_print,
        varname='K_ss',funcname='A_hh-K-B'
    )

