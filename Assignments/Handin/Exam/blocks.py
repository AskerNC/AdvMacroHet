import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def block_pre(par,ini,ss,path,ncols=1):
    for thread in nb.prange(ncols):
            
        # unpack
        A = path.A[thread,:]
        A_hh = path.A_hh[thread,:]
        C_hh = path.C_hh[thread,:]
        clearing_A = path.clearing_A[thread,:]
        clearing_Y = path.clearing_Y[thread,:]
        Gamma = path.Gamma[thread,:]
        I = path.I[thread,:]
        K = path.K[thread,:]
        L = path.L[thread,:]
        r = path.r[thread,:]
        rK = path.rK[thread,:]
        w = path.w[thread,:]
        Y = path.Y[thread,:]


        taua = path.taua[thread,:]
        thetaa = path.thetaa[thread,:]
        xi = path.xi[thread,]
        alpha = path.alpha[thread,:]

        
        K_lag = lag(ini.K,K)
        
        # a. exogenous
        L[:] = 1.0


        # b. Wages and the interest rate 
        w[:]  = (1-alpha)*Gamma * (K_lag/L)**(alpha)
        rK[:] = alpha*Gamma * (K_lag/L)**(alpha-1)
        r[:]  = rK -par.delta 

        # c. Production
        Y[:]  = Gamma*K_lag**alpha *L**(1-alpha)
        
        # d. Capital Law-of-motion
        I[:] = K - (1-par.delta) * K_lag
        A[:] = K

        # e taxes and transfers 
        taua[:]   = 1- (1-ss.taua)   * ( ss.r*ss.K/(r*K_lag))**(par.uptau)
        thetaa[:] = 1- (1-ss.thetaa) * ( ss.r/r)**(par.vartheta)
        # Because r is so small, we need to adjust thetaa when ss.r and r are close to each other to be exactly zero
        I =  np.abs(np.abs(r)-np.abs(ss.r))<par.tol_theta
        
        thetaa[I] = 0.
        
        xi[:] = (taua + thetaa) *r*K_lag


@nb.njit
def block_post(par,ini,ss,path,ncols=1):
    for thread in nb.prange(ncols):
            
        # unpack
        A = path.A[thread,:]
        A_hh = path.A_hh[thread,:]
        C_hh = path.C_hh[thread,:]
        clearing_A = path.clearing_A[thread,:]
        clearing_Y = path.clearing_Y[thread,:]
        Gamma = path.Gamma[thread,:]
        I = path.I[thread,:]
        K = path.K[thread,:]
        L = path.L[thread,:]
        r = path.r[thread,:]
        rK = path.rK[thread,:]
        w = path.w[thread,:]
        Y = path.Y[thread,:]

        taua = path.taua[thread,:]
        thetaa = path.thetaa[thread,:]
        alpha = path.alpha[thread,:]
        

        
        std_ws = path.std_ws[thread,:]
        rA_hh  = path.rA_hh[thread,:]
        std_rA = path.std_rA[thread,:]
        std_A = path.std_A[thread,:]
        
        # There are not multiple threads for D
        D = path.D


        # a. Clearings
        clearing_Y[:] = Y-(C_hh + I )        
        clearing_A[:] = A-A_hh


        # b. interesting outcomes 
        ## This has computational costs for computing jacs, 
        # but I couldn't get to work without loops, and don't have time to find a smarter way
    

        # Fill out with zeros
        
        std_ws[:] = 0.
        rA_hh[:] = 0.
        std_rA[:] = 0. 
        std_A[:] = 0. 
    
        
        # Means
        for t in nb.prange(par.T):
            for i_z in nb.prange(par.Nz):
                r_i  = (1-taua[t]-thetaa[t]) * ((rK[t] *( 1 +  par.chi_grid[i_z]* par.sigma_chi  - (1-par.chi_grid[i_z])* par.sigma_chi* par.pi_chi_obar/par.pi_chi_ubar ) ) -par.delta ) 
                for i_fix in nb.prange(par.Nfix):
                    rA_hh[t] += np.sum( D[t,i_fix,i_z] * (par.a_grid*r_i)  )



        # Std deviations
        for t in nb.prange(par.T):
            for i_z in nb.prange(par.Nz):
                r_i  = (1-taua[t]-thetaa[t] ) * ((rK[t] *( 1 +  par.chi_grid[i_z]* par.sigma_chi  - (1-par.chi_grid[i_z])* par.sigma_chi* par.pi_chi_obar/par.pi_chi_ubar ) ) -par.delta ) 
                for i_fix in nb.prange(par.Nfix):
                    std_ws[t] += np.sum( D[t,i_fix,i_z]* (w[t]*(1- par.s_grid[i_z]) )**2 ) 
                    
                    std_rA[t] += np.sum( D[t,i_fix,i_z] * (par.a_grid*r_i - rA_hh[t] )**2  )

                    std_A[t] += np.sum(D[t,i_fix,i_z] * (par.a_grid-A_hh[t])**2 ) 

            std_ws[t] = np.sqrt(std_ws[t])
            std_rA[t] = np.sqrt(std_rA[t])
            std_A[t] = np.sqrt(std_A[t])

        