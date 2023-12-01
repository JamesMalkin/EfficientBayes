import torch
import math
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from tueplots import bundles
from matplotlib import cm
from tueplots import figsizes, fontsizes, fonts
from matplotlib import rcParams

figsizes.icml2022_full(tight_layout=False)
plt.rcParams.update(bundles.icml2022(nrows=9, ncols=4, usetex=False, column='full'))
rcParams['mathtext.default'] = 'regular'

fig, axs = plt.subplots(9, self.cols, dpi=600, figsize=(5.13,9.9))
color = cm.get_cmap('summer', 5)
powers = np.array([0.5, 2/3, 4/3, 2])
var_list = []
hessian_list = []
lr_list = []
x = []
for i in range(4):
    for j in range(3):
        var_list.append(np.load('data-fig5/var_list_{}.npy'.format(3*i+j)))
        hessian_list.append(np.load('data-fig5/hessian_list_{}.npy'.format(3*i+j)))
        lr_list.append(np.load('data-fig5/lr_list_{}.npy'.format(3*i+j)))
        x.append(np.load('data-fig5/x_{}.npy'.format(3*i+j)))
for n in range(9):
    if n == 0:
        for m in range(4):
            axs[n][m].spines['top'].set_visible(False)
            axs[n][m].spines['right'].set_visible(False)
            axs[n][m].scatter(hessian_list[m*3], (var_list[m*3].flatten()), color=color(m*0.25), s=0.08, alpha=0.8)
            axs[n][0].set_ylabel('$\sigma^{2}$', fontsize=6)
            axs[n][m].set_xscale(value='log')
            axs[n][m].set_yscale(value='log')
            axs[n][m].set_ylim([10**-5.25,10**(-2.5)])
            axs[n][m].set_xlim([10**-7, 10**-1])
       
    if n == 1:
        for m in range(4):
            axs[n][m].spines['top'].set_visible(False)
            axs[n][m].spines['right'].set_visible(False)
            axs[n][m].scatter(hessian_list[m*3+1], (var_list[m*3+1].flatten()), color=color(m*0.25), s=0.08, alpha=0.8)
            axs[n][0].set_ylabel('$\sigma^{2}$', fontsize=6)
            axs[n][m].set_xscale(value='log')
            axs[n][m].set_yscale(value='log')
            axs[n][m].set_ylim([10**-5.25,10**(-2.5)])
            axs[n][m].set_xlim([10**-7, 10**-1])

    if n == 2:
        for m in range(4):
            axs[n][m].spines['top'].set_visible(False)
            axs[n][m].spines['right'].set_visible(False)
            axs[n][m].scatter(hessian_list[m*3+2], (var_list[m*3+2].flatten()), color=color(m*0.25), s=0.08, alpha=0.8)
            axs[n][m].set_xlabel('Hessian', fontsize=6)
            axs[n][0].set_ylabel('$\sigma^{2}$', fontsize=6)
            axs[n][m].set_xscale(value='log')
            axs[n][m].set_yscale(value='log')
            axs[n][m].set_ylim([10**-5.25,10**(-2.5)])
            axs[n][m].set_xlim([10**-7, 10**-1])

    if n == 3:
        for m in range(4):
            axs[n][m].spines['top'].set_visible(False)
            axs[n][m].spines['right'].set_visible(False)
            axs[n][m].scatter(0.0001*(1/(lr_list[m*3]).flatten()), ((var_list[m*3].flatten())), color=color(m*0.25), s=0.08, alpha=0.8)
            axs[n][0].set_ylabel('$\sigma^{2}$', fontsize=6)
            axs[n][m].set_xscale(value='log')
            axs[n][m].set_yscale(value='log')
            axs[n][m].set_xlim([10**-1.9, 10**1.1])
            axs[n][m].set_ylim([10**-5.25,10**(-2.5)])

    if n == 4:
        for m in range(4):
            axs[n][m].spines['top'].set_visible(False)
            axs[n][m].spines['right'].set_visible(False)
            axs[n][m].scatter(0.0001*(1/(lr_list[m*3+1]).flatten()), ((var_list[m*3+1].flatten())), color=color(m*0.25), s=0.08, alpha=0.8)
            axs[n][0].set_ylabel('$\sigma^{2}$', fontsize=6)
            axs[n][m].set_xscale(value='log')
            axs[n][m].set_yscale(value='log')
            axs[n][m].set_xlim([10**-1.9, 10**1.1])
            axs[n][m].set_ylim([10**-5.25,10**(-2.5)])

    if n == 5:
        for m in range(4):
            axs[n][m].spines['top'].set_visible(False)
            axs[n][m].spines['right'].set_visible(False)
            axs[n][m].scatter(0.0001*(1/(lr_list[m*3+2]).flatten()), ((var_list[m*3+2].flatten())), color=color(m*0.25), s=0.08, alpha=0.8)
            axs[n][m].set_xlabel('Learning Rate', fontsize=6)
            axs[n][0].set_ylabel('$\sigma^{2}$', fontsize=6)
            axs[n][m].set_xscale(value='log')
            axs[n][m].set_yscale(value='log')
            axs[n][m].set_ylim([10**-5.25,10**(-2.5)])
            axs[n][m].set_xlim([10**-1.9, 10**1.1])

    if n == 6:
        for m in range(4):
            axs[n][m].spines['top'].set_visible(False)
            axs[n][m].spines['right'].set_visible(False)
            axs[n][m].scatter(np.array(100*((x[m*3]/(20*20)).flatten().tolist())), var_list[m*3].flatten(), color=color(m*0.25), s=0.08, alpha=0.8)
            axs[n][0].set_ylabel('$\sigma^{2}$', fontsize=6)
            axs[n][m].set_xscale(value='log')
            axs[n][m].set_yscale(value='log')
            axs[n][m].set_ylim([10**-5.25,10**(-2.5)])
            axs[n][m].set_xticklabels([6*10**-1, 10**0], minor=True)

    if n == 7:
        for m in range(4):
            axs[n][m].spines['top'].set_visible(False)
            axs[n][m].spines['right'].set_visible(False)
            axs[n][m].scatter(np.array(100*((x[m*3+1]).mean(axis=0).flatten().tolist())), var_list[m*3+1].flatten(), color=color(m*0.25), s=0.08, alpha=0.8)
            axs[n][0].set_ylabel('$\sigma^{2}$', fontsize=6)
            axs[n][m].set_xscale(value='log')
            axs[n][m].set_yscale(value='log')    
            axs[n][m].set_xticklabels([10**-1], minor=True)
            axs[n][m].set_xlim([10**-1.5, 10**-0.5])
            axs[n][m].set_ylim([10**-5.25,10**(-2.5)])

    if n == 8:
        for m in range(4):
            axs[n][m].spines['top'].set_visible(False)
            axs[n][m].spines['right'].set_visible(False)
            axs[n][m].scatter(np.array(10*((x[m*3+2]).mean(axis=0).flatten().tolist())), var_list[m*3+2].flatten(), color=color(m*0.25), s=0.08, alpha=0.8)
            axs[n][m].set_xlabel('Input Rate |x|', fontsize=6)
            axs[n][0].set_ylabel('$\sigma^{2}$', fontsize=6)
            axs[n][m].set_xscale(value='log')
            axs[n][m].set_yscale(value='log')     
            axs[n][m].set_xlim([10**-1.5, 10**-0.5])
            axs[n][m].set_ylim([10**-5.25,10**(-2.5)])
            axs[n][m].set_xticklabels([10**-1], minor=True)
            
