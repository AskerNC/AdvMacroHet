import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem
import blocks

class HANCModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','sim','ss','path']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['rK','w','taua','thetaa','xi'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','u','v'] # outputs
        self.intertemps_hh = ['vbeg','vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['alpha','Gamma'] # exogenous shocks
        self.unknowns = ['K'] # endogenous unknowns
        self.targets = ['clearing_A'] # targets = 0

        # d. all variables
        self.varlist = [
            'A','alpha','clearing_A','clearing_Y',
            'Gamma','I','K','L','r','rK','w','Y',
            'taua','thetaa','xi',
            'std_ws', 'rA_hh','std_rA','std_A', # Interesting means and std variations 
            ]

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.block_pre = blocks.block_pre
        self.block_post = blocks.block_post

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 3 # number of fixed discrete states
        par.Ns = 5
        par.Nchi = 2
        par.Nz = par.Ns*par.Nchi # number of stochastic discrete states

        # a. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta_breve = 0.96 # discount factor, mean
        par.sigma_beta = 0.01 # discount factor, spread
        par.kappa = 0.5 # weight on utility of wealth
        par.a_ubar = 5.0 # luxury of utility of wealth

        # b. income process
        par.rho_s = 0.95 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of persistent shock

        # c. return process
        par.sigma_chi = 0.10 # depreciation rate, spread
        par.pi_chi_obar = 0.5 # depreciation rate, spread
        par.pi_chi_ubar = 0.5 # depreciation rate, spread

        # d. production and investment
        par.alpha_ss = 0.30 # cobb-douglas
        par.delta = 0.10 # depreciation rate, mean in ss
        
        # d.2 taxes 
        par.taua_ss = 0. # Tax rate in ss
        par.uptau = 0.  # Whether to adjust tax rate to keep agg capital income constant (toggle on by 1. )
        
        ## Constant interest rate tax rate 
        par.thetaa_ss = 0.
        par.vartheta = 0.

        # d. grids         
        par.a_max = 100.0 # maximum point in grid for a
        par.Na = 100 # number of grid points

        # e. shocks
        par.jump_Gamma = 0.00 # initial jump
        par.rho_Gamma = 0.0 # AR(1) coefficient

        par.jump_alpha = 0.01 # initial jump
        par.rho_alpha = 0.90 # AR(1) coefficient

        # . misc.
        par.T = 500 # length of transition path        
        par.simT = 2_000 # length of simulation 
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        par.tol_broyden = 1e-12 # tolerance when solving eq. system

        par.tol_theta = 1e-6 # tolerance for r being close enough to r_ss to set theta=theta_ss

        
    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids        
        par.beta_grid = np.zeros(par.Nfix)

        # They need separate grid but the s grid is repeated Nchi times, and the chi_grid is repeated Ns times 
        par.s_grid = np.zeros(par.Nz)
        par.chi_grid= np.tile(np.array([0.,1.]),par.Ns)
        
        # Unconditional likelihood of chi=1
        #par.pi_chi = (1+ par.pi_chi_ubar - par.pi_chi_obar )/2


        # b. transition matrices
        par.s_trans = np.zeros((par.Ns,par.Ns))
        par.chi_trans = np.zeros((par.Nchi,par.Nchi))

        par.z_trans = np.zeros((par.Nz,par.Nz))

        # c. solution
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss