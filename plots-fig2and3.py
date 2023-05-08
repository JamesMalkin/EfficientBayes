import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
from tueplots import bundles
from matplotlib import cm
from tueplots import figsizes, fontsizes, fonts
from matplotlib import rcParams

accs_arr = np.load('accs_arr.npy')
loss_arr = np.load('loss_arr.npy')
rel_arr = np.load('rel_arr.npy')
sig_arr = np.load('sig_arr.npy')

relcost = lambda sig, p, s: (s**p/p)*sig**(-p)

figsizes.icml2022_full(tight_layout=False)

coeffs = 10**(np.arange(-3, 4.24, 0.25, dtype=float))

col = cm.get_cmap('summer', 5)

plt.rcParams.update(bundles.icml2022(nrows=2, ncols=4, usetex=False, column='full'))
rcParams['mathtext.default'] = 'regular'
fig, ax = plt.subplots(2,4, figsize=(5.62,2.7), dpi=600)
for i, p in enumerate(powers):
    ax[0][i].plot((coeffs), accs_arr[i][evens], c=col(i*0.25))
    ax[0][i].plot((coeffs), accs_arr[i][evens==False], color='grey', alpha = 0.6)
    ax[0][i].set_xscale(value='log')
    ax[0][0].set_ylabel('Accuracy %')
    ax[0][i].set_xlim([coeffs[2], coeffs[-6]])
    right_side = ax[0][i].spines["right"]
    right_side.set_visible(False)
    top_side = ax[0][i].spines["top"]
    top_side.set_visible(False)
   
    ax[1][i].plot((coeffs), sig_arr[i][evens], c=col(i*0.25))
    ax[1][i].plot((coeffs), sig_arr[i][evens==False], c='grey', alpha = 0.6)
    ax[1][i].set_xlim([coeffs[2], coeffs[-6]])
    ax[1][i].set_xscale(value='log')
    ax[1][i].set_yscale(value='log')
    ax[1][i].set_xlabel('c')
    ax[1][0].set_ylabel('$\sigma$')
    right_side = ax[1][i].spines["right"]
    right_side.set_visible(False)
    top_side = ax[1][i].spines["top"]
    top_side.set_visible(False)
    

plt.savefig('c-acc')

 
plt.rcParams.update(bundles.icml2022(nrows=3, ncols=4, usetex=False, column='full'))
fig, ax = plt.subplots(3,4, figsize=(5.62,4.1), dpi=600)
for i, p in enumerate(powers):
    ax[0][i].plot(sig_arr[i][evens], accs_arr[i][evens], c=col(i*0.25))
    ax[0][i].plot(sig_arr[i][evens==False], accs_arr[i][evens==False], color='grey', alpha = 0.6)
    ax[0][i].set_xscale(value='log')
    ax[0][i].set_xlabel('$\sigma$')
    ax[0][0].set_ylabel('Accuracy %')
    ax[0][i].set_xlim([10**-1.2, 10**0.5])
    right_side = ax[0][i].spines["right"]
    right_side.set_visible(False)
    top_side = ax[0][i].spines["top"]
    top_side.set_visible(False)
    
    ax[1][i].plot(sig_arr[i][evens], relcost(sig_arr[i][evens], p, 1), c=col(i*0.25))
    ax[1][i].set_xscale(value='log')
    ax[1][i].set_yscale(value='log')
    ax[1][0].set_ylabel('Reliability cost')
    ax[1][i].set_xlabel('$\sigma$')
    ax[1][i].set_ylim([10**-1.5, 10**2])
    ax[1][i].set_xlim([10**-1.2, 10**0.5])
    right_side = ax[1][i].spines["right"]
    right_side.set_visible(False)
    top_side = ax[1][i].spines["top"]
    top_side.set_visible(False)

    ax[2][i].plot(relcost(sig_arr[i][evens], p, 1), accs_arr[i][evens], c=col(i*0.25))
    ax[2][i].plot(relcost(sig_arr[i][evens==False], p, 1), accs_arr[i][evens==False], color='grey', alpha = 0.6)
    ax[2][i].set_xscale(value='log')
    ax[2][i].set_xlabel('Reliability Cost')
    ax[2][0].set_ylabel('Accuracy %')
    ax[2][i].set_xlim([10**-0.3, 10**1])
    right_side = ax[2][i].spines["right"]
    right_side.set_visible(False)
    top_side = ax[2][i].spines["top"]
    top_side.set_visible(False)

    
plt.savefig('var-acc')

