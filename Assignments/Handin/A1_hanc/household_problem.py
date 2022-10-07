import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,taua,taul,vbeg_a_plus,vbeg_a,a,c,ell,u):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """
    
    for i_fix in nb.prange(par.Nfix):

        # a. solve step
        for i_z in range(par.Nz):
            # i. EGM

            # a. prepare
            wt = (1-taul)*w*par.z_grid[i_z]*par.zeta[i_fix]
            rt = (1-taua)* r

            fac = (wt/ par.varphi[i_fix] ) **(1/ par.nu )
            
            # b. use FOCs
            c_endo = ( par.beta * vbeg_a_plus[i_fix,i_z] ) **( -1/ par.sigma )
            ell_endo = fac *( c_endo ) **( - par.sigma / par.nu )
            
            # c. interpolation
            m_endo = c_endo + par.a_grid - wt * ell_endo
            m_exo = (1+ rt ) * par.a_grid
            
            interp_1d_vec( m_endo , c_endo , m_exo , c[i_fix,i_z] )

            interp_1d_vec( m_endo , ell_endo , m_exo , ell[i_fix,i_z] )
            
            a = m_exo + wt * ell - c
            # d. refinement at borrowing constraint
            for i_a in range( par.Na ) :
                if a[i_fix,i_z,i_a ] < 0.0:
                    # i. binding constraint for a
                    a[i_fix,i_z,i_a ] = 0.0
                    # ii. solve FOC for ell
                    elli = ell[i_fix,i_z, i_a ]
                    it = 0
                    while True :
                        ci = (1+ rt ) * par.a_grid [i_a] + wt * elli
                        error = elli - fac * ci **( - par.sigma / par.nu )
                        if np . abs ( error ) < par.tol_ell :
                            break
                        else :
                            derror = 1 - fac *( - par.sigma / par.nu ) * ci**( - par.sigma / par.nu -1) * wt
                            elli = elli - error / derror
                        it += 1
                        
                        if it > par.max_iter_ell : raise ValueError("too many iterations")
                        # iii . save
            
                    c[i_fix,i_z,i_a ] = ci
                    ell[i_fix,i_z,i_a ] = elli
                    u[i_fix,i_z,i_a] = c[i_fix,i_z,i_a]**(1-par.sigma)/(1-par.sigma)-par.varphi[i_fix] *ell[i_fix,i_z,i_a]**(1+par.nu)/(1+par.nu)



        # b. expectation step
        v_a = (1+rt)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

