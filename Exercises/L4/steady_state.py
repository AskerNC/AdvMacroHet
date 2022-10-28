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
    #par.a_grid[:] = equilogspace(0.0,ss.w*par.a_max,par.Na)
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)

    # b. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    ss.z_trans[0,:,:] = z_trans
    ss.Dz[0,:] = z_ergodic
    ss.Dbeg[0,:,0] = ss.Dz[0,:] # ergodic at a_lag = 0.0
    ss.Dbeg[0,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # a. raw value
    y = (1-par.tau)*par.z_grid
    c = m = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    v_a = (1+ss.r)*c**(-par.sigma)

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans@v_a

    
def obj_ss(B_ss,model,do_print=False):
    """
    > The function `obj_ss` takes a guess for the steady state capital stock, and then computes the
    implied steady state interest rate, and then solves the household problem, and then computes the
    implied aggregate savings, and then returns the difference between the implied aggregate savings and
    the steady state capital stock
    
    Args:
      B_ss: the steady state capital stock
      model: the model object
      do_print: print the results of the optimization. Defaults to False
    
    Returns:
      The difference between the aggregate assets and the aggregate debt.
    """

    par = model.par
    ss = model.ss

    # a. production
    #ss.Gamma = par.Gamma_ss # model user choice
    #ss.K = K_ss
    ss.B  = B_ss
    #ss.L = 1.0 # by assumption
    #ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)    
    ss.Y  = np.sum(ss.D*par.z_grid[:,np.newaxis])
    # b. implied prices
    #ss.rk = par.alpha*ss.Gamma*(ss.K/ss.L)**(par.alpha-1.0)
    # Using distribution of productivity
    
    ss.r =  1/ss.B *par.tau*ss.Y 
    
    if np.isclose(par.tau,0):
        ss.r= 0.

    #ss.w = (1.0-par.alpha)*ss.Gamma*(ss.K/ss.L)**par.alpha

    # c. household behavior
    if do_print:

        print(f'guess {ss.B = :.4f}')    
        print(f'implied {ss.r = :.4f}')

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    ss.A_hh = np.sum(ss.a*ss.D) # hint: is actually computed automatically
    ss.C_hh = np.sum(ss.c*ss.D)

    if do_print: print(f'implied {ss.A_hh = :.4f}')

    # d. bond market clearing
    ss.clearing_A = ss.A_hh-ss.B

    ss.I = ss.B*ss.r - par.tau* ss.Y
    ss.C = ss.Y - ss.I
    ss.clearing_C = ss.C_hh-ss.C

    return ss.clearing_A # target to hit
    
def find_ss(model,method='direct',do_print=False,K_min=0.01,K_max=25.0,NK=100):
    '''It finds the steady state of a model by either the direct or indirect method
    
    Parameters
    ----------
    model
        the model object
    method, optional
        'direct' or 'indirect'
    do_print, optional
        whether to print the results
    K_min
        minimum value of K to search for steady state
    K_max
        the maximum value of capital to search for the steady state
    NK, optional
        number of points to use in the grid search
    
    '''
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    if method == 'direct':
        find_ss_direct(model,do_print=do_print,K_min=K_min,K_max=K_max,NK=NK)
    elif method == 'indirect':
        find_ss_indirect(model,do_print=do_print)
    else:
        raise NotImplementedError

    if do_print: print(f'found steady state in {elapsed(t0)}')


# K should be renamed to B
def find_ss_direct(model,do_print=False,K_min=0,K_max=10.0,NK=10):
    """
     find steady state using direct method 
    It finds the steady state by first doing a broad search for the steady state capital stock, then it
    narrows the search interval, and finally it uses Brent's method to find the steady state capital
    stock
    
    :param model: the model object
    :param do_print: whether to print the results of the search, defaults to False (optional)
    :param K_min: minimum value of capital stock to try, defaults to 0 (optional)
    :param K_max: the maximum value of capital stock to try
    :param NK: number of trial values for K_ss, defaults to 10 (optional)
    """
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
    
    K_max, K_min = find_K_interval(K_ss_vec,clearing_A,model,do_print)


    if do_print: print(f'K in [{K_min:12.8f},{K_max:12.8f}]\n')

    # c. search
    if do_print: print(f'### step 3: search ###\n')

    root_finding.brentq(
        obj_ss,K_min,K_max,args=(model,),do_print=do_print,
        varname='K_ss',funcname='A_hh-K'
    )

def find_ss_indirect(model,do_print=False):
    """ find steady state using indirect method """
    
    
    raise NotImplementedError("Indirect method not implemented for bond model")

    par = model.par
    ss = model.ss

    # a. exogenous and targets
    #ss.L = 1.0
    ss.r = par.r_ss_target
    #ss.w = par.w_ss_target

    assert (1+ss.r)*par.beta < 1.0, '(1+r)*beta < 1, otherwise problems might arise'

    # b. stock and capital stock from household behavior
    model.solve_hh_ss(do_print=do_print) # give us ss.a and ss.c (steady state policy functions)
    model.simulate_hh_ss(do_print=do_print) # give us ss.D (steady state distribution)
    if do_print: print('')

    ss.B = ss.A_hh = np.sum(ss.a*ss.D)
    
    # c. back technology and depreciation rate
    #ss.Gamma = ss.w / ((1-par.alpha)*(ss.K/ss.L)**par.alpha)
    #ss.rk = par.alpha*ss.Gamma*(ss.K/ss.L)**(par.alpha-1)
    #par.delta = ss.rk - ss.r
    # Back out tau
    ss.Y = np.sum(ss.D*par.z_grid[:,np.newaxis])

    par.tau = ss.r*ss.B / ss.Y

    # d. remaining
    #ss.C = ss.Y - par.delta*ss.K
    #ss.C_hh = np.sum(ss.D*ss.c)

    ss.I = ss.B*ss.r - par.tau* ss.Y
    ss.C = ss.Y - ss.I
    ss.C_hh = np.sum(ss.D*ss.c)

    # e. print
    if do_print:

        print(f'Implied B = {ss.B:6.3f}')
        print(f'Implied Y = {ss.Y:6.3f}')
        print(f'Implied tau = {par.tau:6.3f}') # check is positive
        print(f'Implied B/Y = {ss.B/ss.Y:6.3f}') 
        print(f'Discrepancy in b-A_hh = {ss.B-ss.A_hh:12.8f}') # = 0 by construction
        print(f'Discrepancy in C-C_hh = {ss.C-ss.C_hh:12.8f}\n') # != 0 due to numerical error 




def find_K_interval(K_ss_vec,clearing_A,model,do_print):
    '''
    This function makes sure the interval for K have a negative and 
    a postive value for clearing_A
    '''
    lower_bound = (np.sum((clearing_A<0))<1)
    K_max =  np.min(K_ss_vec[clearing_A < 0])
    k_upper = np.max(K_ss_vec)

    while lower_bound:
        k_upper += 1
        new_clearing =  obj_ss(k_upper,model,do_print=do_print)
        if do_print:
            print(f'The proposed K_max was to small trying  {k_upper} ->')
            print(f'clearing_A = {new_clearing:12.8f}\n')

        if np.notna(new_clearing) and new_clearing<0:
            K_max = k_upper
            


    upper_bound = (np.sum((clearing_A>0))<1)
    K_min =  np.max(K_ss_vec[clearing_A > 0])
    k_lower = np.min(K_ss_vec)

    while upper_bound:
        k_lower -= 1
        new_clearing =  obj_ss(k_lower,model,do_print=do_print)

        if do_print:
            print(f'The proposed K_min was to large trying  {k_lower} ->')
            print(f'clearing_A = {new_clearing:12.8f}\n')

        if np.notna(new_clearing) and new_clearing<0:
            k_min = k_lower
            
    return K_max, K_min