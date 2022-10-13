import time
import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

import root_finding
from scipy import optimize


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

def obj_ss(x ,model,do_print=False):
    """ objective when solving for steady state capital and labor """
    K_ss = x[0]
    L_ss = x[1]

    par = model.par
    ss = model.ss

    # a. production
    ss.Gamma = par.Gamma_ss # model user choice
    ss.K = K_ss
    ss.L = L_ss # by assumption
    ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)    

    # b. implied prices
    ss.rk = par.alpha*ss.Gamma*(ss.K/ss.L)**(par.alpha-1.0)
    ss.r = ss.rk - par.delta

    if np.isclose(ss.r,0):
        # Model cant be sovled for r close to zero 
        print('Tried r close to zero, returning the previous clearing values')
        return np.array([ss.clearing_A,ss.clearing_L])
    
    ss.w = (1.0-par.alpha)*ss.Gamma*(ss.K/ss.L)**par.alpha

    # c. household behavior
    if do_print:

        print(f'guess {ss.K = :.4f}')    
        print(f'implied {ss.r = :.4f}')
        print(f'implied {ss.w = :.4f}')

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    #ss.A_hh = np.sum(ss.a*ss.D) # hint: is actually computed automatically
    #ss.C_hh = np.sum(ss.c*ss.D)
    #ss.U_hh = np.sum(ss.u*ss.D)
    #ss.ELL_hh = np.sum( ss.ell*ss.D)
    ss.L_hh   = np.sum( par.zeta_grid@(par.z_grid@(ss.ell*ss.D)) )
    #ss.L_hh = np.sum(ss.ell * par.z_grid[np.newaxis,:,np.newaxis] * par.zeta_grid[:,np.newaxis,np.newaxis] * ss.D)

    if do_print: print(f'implied {ss.A_hh = :.4f}')

    # d. bonds market and taxe income

    ss.taxa =  ss.taua*ss.r*ss.A_hh
    ss.taxl =   ss.taul*ss.w*ss.L_hh
    ss.B = -1/ss.r * ( ss.G -ss.taxa -ss.taxl)

    if do_print: print(f'implied {ss.B = :.4f}')

    # e. market clearing

    # Assets clearing
    ss.clearing_A = ss.A_hh-ss.K-ss.B
    
    # Labor market clearing 
    ss.clearing_L = ss.L_hh-ss.L


    # goods clearing 
    ss.I = ss.K - (1-par.delta)*ss.K
    ss.IB = ss.r*ss.B+ss.G-ss.taxa -ss.taxl
    ss.C = ss.Y - ss.I - ss.IB - ss.G

    ss.clearing_C = ss.C_hh-ss.C

    return np.array([ss.clearing_A,ss.clearing_L]) # target to hit
    
def find_ss(model,method='direct',do_print=False,
                    x0=np.array([1.,2.]),bounds= ((0.1,10),(0.1,10)),root_method='hybr',
                    N=5,
                    solveclearing='A',roption='positive',lower=0.5,upper_mult=6,step=0.05,kl_bounds = None):
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    if method == 'direct':
        find_ss_direct(model,bounds,N,do_print=do_print)

    elif method == 'root':
        find_ss_root(model,x0,root_method,do_print=do_print)
    elif method=='kl':
        find_ss_kl(model,solveclearing,roption,lower,upper_mult,step, kl_bounds,do_print=do_print)
    else:
        raise NotImplementedError

    if do_print: print(f'found steady state in {elapsed(t0)}')

def find_ss_direct(model,bounds,N,do_print=False):
    """ find steady state using direct method """
    raise NotImplementedError

    # a. broad search
    if do_print: print(f'### step 1: broad search ###\n')

    K_ss_vec = np.linspace(bounds[0,0],bounds[0,1],N) # trial values
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


def find_ss_root(model,x0,root_method, do_print=False):
    if do_print: print(f'### step 1 going strait to optimize.root with guess (K,L) = ({x0[0]:.2f},{x0[1]:.2f})')

    res = optimize.root(obj_ss,x0=x0,args=(model,) ,method=root_method,tol=model.par.tol_root)

    if do_print:
        print(res)



def find_ss_kl(model,solveclearing,roption,lower,upper_mult,step,kl_bounds,do_print=False):
    # Gamma for finding klguess0
    model.ss.Gamma = model.par.Gamma_ss

    klguess0 = (model.par.delta/(model.par.alpha*model.ss.Gamma))**(1/(model.par.alpha-1))
    if roption=='positive': 

        if kl_bounds is None:
            kl_bounds = [lower,klguess0*(1-step)]

        root_finding.brentq(
            obj_ss_kl,kl_bounds[0],kl_bounds[1],args=(model,solveclearing),do_print=do_print,
            varname='KL',funcname=f'{solveclearing}_clearing'
            )

    elif roption=='negative':
        if kl_bounds is None:
            kl_bounds = [klguess0*(1+step),klguess0*upper_mult]
        
        fa = obj_ss_kl(kl_bounds[0],model,solveclearing)
        fb = obj_ss_kl(kl_bounds[1],model,solveclearing)

        assert fa*fb<=0, f'\nSolution not found for kl bounds = {kl_bounds[0]:10.3f}, {kl_bounds[1]:10.3f}\nClearings:                       = {fa:10.3f}, {fb:10.3f}'

        root_finding.brentq(
            obj_ss_kl,kl_bounds[0],kl_bounds[1],args=(model,solveclearing),do_print=do_print,
            varname='KL',funcname=f'{solveclearing}_clearing'
            )
    else:
        raise NotImplementedError


def obj_ss_kl(x ,model,solveclearing,do_print=False):
    """ objective when solving for steady state capital-labor ratio """
    par = model.par
    ss = model.ss

    ss.KL = x
    
    # a. production
    ss.Gamma = par.Gamma_ss # model user choice
    #ss.K = K_ss
    #ss.L = L_ss # by assumption
    #ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)    

    # b. implied prices
    ss.rk = par.alpha*ss.Gamma*(ss.KL)**(par.alpha-1.0)
    ss.r = ss.rk - par.delta

    if np.isclose(ss.r,0):
        # Model cant be sovled for r close to zero 
        print('Tried r too close to zero, returning the previous clearing values')
        return ss.clearing_A
    
    ss.w = (1.0-par.alpha)*ss.Gamma*(ss.KL)**par.alpha

    # c. household behavior
    if do_print:

        print(f'guess {ss.KL = :.4f}')    
        print(f'implied {ss.r = :.4f}')
        print(f'implied {ss.w = :.4f}')

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    #ss.A_hh = np.sum(ss.a*ss.D) # hint: is actually computed automatically
    #ss.C_hh = np.sum(ss.c*ss.D)
    #ss.U_hh = np.sum(ss.u*ss.D)
    #ss.ELL_hh = np.sum( ss.ell*ss.D)
    ss.L_hh   = np.sum( par.zeta_grid@(par.z_grid@(ss.ell*ss.D)) )
    #ss.L_hh = np.sum(ss.ell * par.z_grid[np.newaxis,:,np.newaxis] * par.zeta_grid[:,np.newaxis,np.newaxis] * ss.D)

    if do_print: print(f'implied {ss.A_hh = :.4f}')

    # d. bonds market and taxe income

    ss.taxa =  ss.taua*ss.r*ss.A_hh
    ss.taxl =   ss.taul*ss.w*ss.L_hh
    ss.B = -1/ss.r * ( ss.G -ss.taxa -ss.taxl)

    if do_print: print(f'implied {ss.B = :.4f}')

    # e. market clearing

    if solveclearing=='A':
        # Solve for assets clearing

        # Assume labor market clearing 
        ss.L = ss.L_hh
        ss.clearing_L = ss.L_hh-ss.L
        # Implied K and Y:
        ss.K = (par.alpha*ss.Gamma/ss.rk)**(1/(1-par.alpha)) * ss.L
        ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)    

        # Assets clearing
        ss.clearing_A = ss.A_hh-ss.K-ss.B

        # goods clearing 
        ss.I = ss.K - (1-par.delta)*ss.K
        ss.IB = ss.r*ss.B+ss.G-ss.taxa -ss.taxl
        ss.C = ss.Y - ss.I - ss.IB - ss.G

        ss.clearing_C = ss.C_hh-ss.C

        return ss.clearing_A # target to hit
    elif solveclearing=='L':
        # Solve for labor market clearing
        # Assume assets clearing
        ss.K = ss.A_hh-ss.B
        # Assets clearing
        ss.clearing_A = ss.A_hh-ss.K-ss.B
        # Implied L and Y:
        ss.L = (ss.rk/(par.alpha*ss.Gamma))**(1/(1-par.alpha)) * ss.K
        ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)    
        
        # Labor market clearing
        ss.clearing_L = ss.L_hh-ss.L

        # goods clearing 
        ss.I = ss.K - (1-par.delta)*ss.K
        ss.IB = ss.r*ss.B+ss.G-ss.taxa -ss.taxl
        ss.C = ss.Y - ss.I - ss.IB - ss.G

        ss.clearing_C = ss.C_hh-ss.C

        return ss.clearing_L

    else: 
        # I think one could also just for the capital labor ratio implied by the household problem is equal to the one implied by the firms
        raise NotImplementedError