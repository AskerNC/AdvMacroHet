import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def block_pre(par,ini,ss,path,ncols=1):

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        chi = path.chi[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        G = path.G[ncol,:]
        Gamma = path.Gamma[ncol,:]
        i = path.i[ncol,:]
        L = path.L[ncol,:]
        NKWC_res = path.NKWC_res[ncol,:]
        pi_w = path.pi_w[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        q = path.q[ncol,:]
        ra = path.ra[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]

        #################
        # implied paths #
        #################


        Gamma_lag       = lag(ss.Gamma,Gamma)
        B_lag           = lag(ss.B,B)
        
        # wages
        w[:] = Gamma 

        # inflation
        pi[:] = (1+pi_w)/(Gamma/Gamma_lag)-1

        

        # Monetary policy
        for t in range(par.T):
            i_lag = i[t-1] if t>0 else ini.i
            i[t] = (1+i_lag)**par.rho_i * ((1+ss.r)*(1+pi[t])**par.phi_pi)**(1-par.rho_i)-1


        pi_lead         = lead(pi,ss.pi)
        # fisher
        r[:] = (1+i)/(1+pi_lead)-1
        ra[:] = lag(ss.r,r)
        
        # q by loop 
        for t in range(par.T):
            q_lag = q[t-1] if t>0 else ini.q
            q[t] = ((1+ra[t])*q_lag -1)/par.delta 

        
        tau[:] = ss.tau+par.omega*ss.q* (B_lag-ss.B)/ss.Y

        # Production
        Y[:] = 1/tau *( (1+q*par.delta)*B_lag- q*B+G+chi)

        # Implied labor 
        L[:] = Y/Gamma

        A[:] = q*B


@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        chi = path.chi[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        G = path.G[ncol,:]
        Gamma = path.Gamma[ncol,:]
        i = path.i[ncol,:]
        L = path.L[ncol,:]
        NKWC_res = path.NKWC_res[ncol,:]
        pi_w = path.pi_w[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        q = path.q[ncol,:]
        ra = path.ra[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]
        
        #################
        # check targets #
        #################
        pi_w_lead = lead(pi_w,ss.pi_w)

        NKWC_res[:] = par.kappa*(par.varphi*(L)**par.nu -1/par.mu *(1-tau)* w*(C_hh)**(-par.sigma)) +par.beta*pi_w_lead -pi_w
        clearing_A[:] = A-A_hh
        clearing_Y[:] = ss.Y-ss.C_hh -ss.G