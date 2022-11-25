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
        NKWPC_res = path.NKWPC_res[ncol,:]
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


        Gamma_lag       = lag(ini.Gamma,Gamma)
        B_lag           = lag(ini.B,B)

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
        


        # Reverse loop to get q
        for t_ in range(par.T):
            t = (par.T-1)-t_
            q_lead = q[t+1] if t < par.T-1 else ss.q
            q[t] = (1+par.delta*q_lead)/(1+r[t])
        

        q_lag = lag(ini.q,q)
        ra[:] = (1+par.delta*q)/q_lag-1


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
        NKWPC_res = path.NKWPC_res[ncol,:]
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

        # The residual of the New Keynesian Phillips Curve.
        NKWPC_res[:] = pi_w -(par.kappa*(par.varphi*(L)**par.nu -1/par.mu *(1-tau)* w*(C_hh)**(-par.sigma)) +par.beta*pi_w_lead )
        clearing_A[:] = A-A_hh
        clearing_Y[:] = ss.Y-ss.C_hh -ss.G