import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def block_pre(par,ini,ss,path,ncols=1):
    """ evaluate transition path - before household block """

    # par, ini, ss, path are namespaces
    # ncols specifies have many versions of the model to evaluate at once
    #   path.VARNAME have shape=(len(unknowns)*par.T,par.T)
    #   path.VARNAME[0,t] for t in [0,1,...,par.T] is always used outside of this function

    for thread in nb.prange(ncols):
        
        # unpack
        A = path.A[thread,:]
        A_hh = path.A_hh[thread,:]
        C = path.C[thread,:]
        C_hh = path.C_hh[thread,:]
        clearing_A = path.clearing_A[thread,:]
        clearing_C = path.clearing_C[thread,:]
        Gamma = path.Gamma[thread,:]
        K = path.K[thread,:]
        L = path.L[thread,:]
        r = path.r[thread,:]
        rk = path.rk[thread,:]
        w = path.w[thread,:]
        Y = path.Y[thread,:]
        B = path.B[thread,:]
        tau = path.tau[thread,:]
        G = path.G[thread,:]

        #################
        # implied paths #
        #################

        # lags and leads of unknowns and shocks
        K_lag = lag(ini.K,K) # copy, same as [ini.K,K[0],K[1],...,K[-2]]
        
        
        # example: K_lead = lead(K,ss.K) # copy, same as [K[1],K[1],...,K[-1],ss.K]

        # Bonds, since government budget is balanced B is the same in every period
        B[:] = ss.B
        

        # VARNAME is used for reading values
        # VARNAME[:] is used for writing in-place

        # a. exogenous
        L[:] = 1.0

        # b. implied prices (remember K is input -> K_lag is known)
        rk[:] = par.alpha*Gamma*(K_lag/L)**(par.alpha-1.0)
        r[:] = rk-par.delta
        w[:] = (1.0-par.alpha)*Gamma*(rk/(par.alpha*Gamma))**(par.alpha/(par.alpha-1.0))

        #b2 implied tax rate:
        tau[:] = (r*B+G)       


        # c. production and consumption
        Y[:] = Gamma*K_lag**(par.alpha)*L**(1-par.alpha)
        C[:] = Y-(K-K_lag)-par.delta*K_lag -G

        # d. total assets
        A[:] = K + B

@nb.njit
def block_post(par,ini,ss,path,ncols=1):
    """ evaluate transition path - after household block """

    for thread in nb.prange(ncols):

        # unpack
        A = path.A[thread,:]
        A_hh = path.A_hh[thread,:]
        C = path.C[thread,:]
        C_hh = path.C_hh[thread,:]
        clearing_A = path.clearing_A[thread,:]
        clearing_C = path.clearing_C[thread,:]
        Gamma = path.Gamma[thread,:]
        K = path.K[thread,:]
        L = path.L[thread,:]
        r = path.r[thread,:]
        rk = path.rk[thread,:]
        w = path.w[thread,:]
        Y = path.Y[thread,:]
        B = path.B[thread,:]
        tau = path.tau[thread,:]
        G = path.G[thread,:]

        ###########
        # targets #
        ###########

        clearing_A[:] = K+B-A_hh
        clearing_C[:] = C-C_hh            