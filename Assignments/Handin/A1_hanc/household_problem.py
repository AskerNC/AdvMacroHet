import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec
from numba import njit, boolean, int32, double, void


@nb.njit(parallel=False)        
def solve_hh_backwards(par,z_trans,r,w,taua,taul,vbeg_a_plus,vbeg_a,a,c,ell,u):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """
    
    # prepare
    rt = (1-taua)* r
    m_exo = (1+ rt ) * par.a_grid
    

    for i_fix in range(par.Nfix):
        # a. solve step
        for i_z in range(par.Nz):
            # i. EGM

            # a. prepare
            wt = w*par.z_grid[i_z]*par.zeta_grid[i_fix]

            fac = (par.theta* wt**par.theta* (1-taul) / ( par.varphi_grid[i_fix] *par.xh_theta )) **(1/ (par.nu-par.theta+1) )
            
            # b. use FOCs
            c_endo = ( par.beta * vbeg_a_plus[i_fix,i_z] ) **( -1/ par.sigma )
            ell_endo = fac *( c_endo ) **( - par.sigma /( par.nu-par.theta+1) )
            
            # c. interpolation
            m_endo = c_endo + par.a_grid - (1-taul)*((wt * ell_endo)**(par.theta)/par.xh_theta )
            
            interp_1d_vec( m_endo , c_endo , m_exo , c[i_fix,i_z] )

            interp_1d_vec( m_endo , ell_endo , m_exo , ell[i_fix,i_z] )
            
            a[i_fix,i_z,:] = m_exo +  (1-taul)*((wt * ell[i_fix,i_z])**(par.theta)/par.xh_theta ) - c[i_fix,i_z]
            
            # d. refinement at borrowing constraint
            for i_a in range( par.Na ):
                if a[i_fix,i_z,i_a ] < 0.0:
                    # i. binding constraint for a
                    a[i_fix,i_z,i_a ] = 0.
                    # ii. solve FOC for ell
                    elli = ell[i_fix,i_z, i_a ]
                    it = 0
                    while True :
                        ci = (1+ rt ) * par.a_grid[i_a] + (1-taul)*((wt * elli)**(par.theta)/par.xh_theta )
                        error = elli - fac * ci **( - par.sigma / (par.nu-par.theta+1) )
                        if np.abs( error ) < par.tol_ell :
                            break
                        else :
                            derror = 1 - fac *( - par.sigma / (par.nu-par.theta+1) ) * ci**( - par.sigma / (par.nu-par.theta+1)-1) * par.theta* wt**par.theta* (1-taul)/par.xh_theta * elli**(par.theta-1) 
                            elli = elli - error / derror
                        it += 1
                        
                        if it > par.max_iter_ell : raise ValueError("too many iterations")
                        # iii . save
            
                    c[i_fix,i_z,i_a] = ci
                    ell[i_fix,i_z,i_a] = elli
        
        
        
        u[i_fix,:,:] = c[i_fix]**(1-par.sigma)/(1-par.sigma)-par.varphi_grid[i_fix]*ell[i_fix]**(1+par.nu)/(1+par.nu)

        # b. expectation step
        v_a = (1+rt)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

