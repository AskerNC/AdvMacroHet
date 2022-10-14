
import time
import pickle
import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt   
plt.style.use('seaborn-whitegrid')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
from matplotlib import cm # for colormaps
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm # for progress bar
from steady_state import obj_ss_kl
import pandas as pd

from HANCModel import HANCModelClass

## The functions used in results.ipynb for showing results 


def clearing_across_kl(model,start,end,N,solveclearing='A',varlist =[ 'Y', 'C_hh','U_hh','A_hh','ELL_hh','I','K','L','B','taxa','taxl','r','w','clearing_A','clearing_C','clearing_L']):
    '''
    Calculates assets clearing across k/l ratio
    '''
    kl_list = np.linspace(start,end,N)
    
    out_dict = {var :np.empty(N) for var in varlist}

    model_calib = model.copy()
    for i in range(N):
        clearingi = obj_ss_kl(kl_list[i],model_calib,solveclearing)
        for key, val in out_dict.items():
            val[i] = getattr(model_calib.ss,key)
                
    return kl_list, out_dict

def plot_clearing_across_kl(kl_list,out_dict,plotvar,solveclearing='A'):
    '''
    Plots assets clearing across k/l ratio
    '''
    fig = plt.figure(figsize=(12,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(kl_list,out_dict[plotvar])
    ax.set_xlabel('$\\frac{K}{L}$-ratio')
    ax.set_ylabel(f'{plotvar}')
    return fig










#### the equilibrium across tau 

def make_model_dict(model,gridsize,start,end,roption='positive',method='kl',
                    varlist =[ 'Y', 'C_hh','U_hh','A_hh','ELL_hh','I','K','L','KL','B','taxa','taxl','r','w']):
    
    '''
    Creates a dict of values of varlist for the equilibria of the model across different taus

    '''


    tau_list = np.linspace(start,end,gridsize)

    
    # Instead store output
    model_dict = {var :np.empty( (gridsize,gridsize)) for var in varlist}

    model_solved= np.full((gridsize,gridsize), True)


    for ia, taua in enumerate(tqdm(tau_list)):
        for il, taul in enumerate(tau_list):
            
            
            # reset tax rates:  
            ss =model.ss
            ss.taua = taua 
            ss.taul = taul

            # Solve model 
            try:
                model.find_ss(method=method,roption=roption,do_print=False)
                # Store welfare
                
                for key, val in model_dict.items():
                    val[ia,il] = getattr(ss,key)
                

            except:
                # Note that model is not solved
                model_solved[ia,il] = False

                for key, val in model_dict.items():
                    val[ia,il] = np.nan



    # store tau lists for plotting 

    model_dict['taua_list'] = tau_list
    model_dict['taul_list'] = tau_list
    model_dict['model_solved'] = model_solved

    return model_dict



def plot_over_taugrid(model_dict,zvar_dict={'U_hh':'Average utility','C_hh':'Average consumption','ELL_hh':'Average labor, $\\ell$','K':'K','r':'r','B':'Bonds'},rows=3,cols=2,figsize=(16,12)):
    '''
    Plot the model dict from the function above across taus
    '''
    
    taua_list = model_dict['taua_list']
    taul_list = model_dict['taul_list']

    taua_grid, taul_grid = np.meshgrid(taua_list,taul_list,indexing='ij')

    if len(zvar_dict)==1:
        rows = cols = 1
    
    
    # a. actual plot
    fig = plt.figure(figsize=figsize,dpi=100)

    for i, item in enumerate(zvar_dict.items()):
        key = item[0]
        value = item[1]
        ax = fig.add_subplot(rows,cols,i+1,projection='3d')
        
        ax.plot_surface(taua_grid,taul_grid,model_dict[key],cmap=cm.jet)
        
        ax.scatter(taua_grid,taul_grid,model_dict[key],cmap=cm.jet)
        
        # b. add labels
        ax.set_xlabel('$\\tau^{a}$') 
        ax.set_ylabel('$\\tau^{\ell}$')
        ax.set_zlabel(value)

        # c. invert xaxis to bring Origin in center front
        ax.invert_xaxis()
        ax.view_init( azim=150,elev=20)
    fig.tight_layout()
    return fig
    