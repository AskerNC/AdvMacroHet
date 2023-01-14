import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit     
def solve_hh_backwards(par,z_trans,rK,w,taua,thetaa,xi,vbeg_plus,vbeg_a_plus,vbeg,vbeg_a,a,c,u,v):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    v_a = np.zeros((par.Nfix,par.Nz,par.Na))

    for i_fix in nb.prange(par.Nfix):

        # a. solve step
        for i_z in nb.prange(par.Nz):
        

            rK_i = rK *( 1 +  par.chi_grid[i_z]* par.sigma_chi  - (1-par.chi_grid[i_z])* par.sigma_chi* par.pi_chi_obar/par.pi_chi_ubar ) 
            r_i =  (1-taua-thetaa) * ( rK_i-par.delta ) 

            # i. EGM
            future = par.beta_grid[i_fix]*vbeg_a_plus[i_fix,i_z]
            
            c_endo = (future + par.kappa* (par.a_grid+par.a_ubar )**(-par.sigma ) ) **(-1/par.sigma)
            
            
            m_endo = c_endo + par.a_grid

            
            
            # ii. interpolation to fixed grid
            m = (1+r_i)*par.a_grid + w*par.s_grid[i_z] + xi

            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            c[i_fix,i_z] = m-a[i_fix,i_z]

            # iii. utility
            v_a[i_fix,i_z] = (1+r_i)*c[i_fix,i_z]**(-par.sigma)

            u[i_fix,i_z] = c[i_fix,i_z]**(1-par.sigma)/(1-par.sigma) + par.kappa *  (a[i_fix,i_z] + par.a_ubar )**(1-par.sigma ) /  (1-par.sigma)
            v[i_fix,i_z] = u[i_fix,i_z] + par.beta_grid[i_fix]*vbeg_plus[i_fix,i_z]

        # b. expectation step
        vbeg[i_fix] = z_trans[i_fix]@v[i_fix]
        vbeg_a[i_fix] = z_trans[i_fix]@v_a[i_fix]
