import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import cv2
from kymatio import Scattering2D
from PIL import Image
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
import scipy
from scipy.ndimage import gaussian_filter1d

try:
    xrange(1)
except NameError:
    xrange = range



matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["figure.figsize"] = [10.0,8.0]
axislabelfontsize= 54

matplotlib.mathtext.rcParams['legend.fontsize']=20


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


plt.rc("axes", linewidth=2.0)
plt.rc("lines", markeredgewidth=3)
plt.rc('axes', labelsize=32)
plt.rc('xtick', labelsize = 32)
plt.rc('ytick', labelsize = 32)

fig_width_pt = 1000 #513.17           # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean=0.9
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]

params = {'backend': 'pdf',
             'axes.labelsize': 54,
             'lines.markersize': 4,
             'font.size': 100,
             'xtick.major.size':6,
             'xtick.minor.size':3,
             'ytick.major.size':6,
             'ytick.minor.size':3,
             'xtick.major.width':0.5,
             'ytick.major.width':0.5,
             'xtick.minor.width':0.5,
             'ytick.minor.width':0.5,
             'lines.markeredgewidth':1,
             'axes.linewidth':1.2,
             'xtick.labelsize': 32,
             'ytick.labelsize': 32,
             'savefig.dpi':2000,
   #      'path.simplify':True,
         'font.family': 'serif',
         'font.serif':'Times',
             'text.usetex':True,
             'text.latex.preamble': [r'\usepackage{amsmath}'],
             'figure.figsize': fig_size}



parameter=['Ns', '$H_s$', '$H_A$' ,'As']

F_wst100_100 = np.loadtxt('st_100_100.txt')
F_wst1_100  =np.loadtxt('st_10_100.txt')
F_wst2_100  =np.loadtxt('st_1_100.txt')
F_wst10_100  =np.loadtxt('st_2_100.txt')
F_wst5_100  =np.loadtxt('st_5_100.txt')

F_ps5_100 = np.loadtxt('ps_5_100.txt')
F_ps100_100 = np.loadtxt('ps_100_100.txt')
F_ps1_100  =np.loadtxt('ps_10_100.txt')
F_ps2_100  =np.loadtxt('ps_1_100.txt')
F_ps10_100  =np.loadtxt('ps_2_100.txt')

wst= [F_wst2_100[0],F_wst10_100[0],F_wst100_100[0]]
ps= [F_ps2_100[0],F_ps10_100[0],F_ps100_100[0]]
noise_amp=[2,10,100]
plt.plot(noise_amp, wst,label='WST', linewidth=4)
plt.plot(noise_amp, ps, label='Power Spectrum',linewidth=4)
plt.ylabel(r'log$_{10}$F$_{ij}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Noise amplitude')
plt.legend()
plt.tight_layout()
plt.savefig('noise.pdf')


F_wst1_90 = np.loadtxt('st_1_90.txt')
F_wst1_100  =np.loadtxt('st_1_100.txt')
F_wst1_120  =np.loadtxt('st_1_120.txt')


F_ps1_90 = np.loadtxt('ps_1_90.txt')
F_ps1_100 = np.loadtxt('ps_1_100.txt')
F_ps1_120  =np.loadtxt('ps_1_120.txt')


wst= [F_wst1_90[0],F_wst1_100[0],F_wst1_120[0]]
ps= [F_ps1_90[0],F_ps1_100[0],F_ps1_120[0]]
res=[ 90,100,120]
plt.plot(res, wst,label='WST', linewidth=4)
plt.plot(res, ps, label='Power Spectrum',linewidth=4)
plt.ylabel(r'log$_{10}$F$_{ij}$')
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Resolution(kms)')
plt.legend()
plt.tight_layout()
plt.savefig('res.pdf')


#Checking whether scatter coefficients actually contain cosmological information

L = 1
J = 2
scattering = Scattering2D(J=J, shape=(32000, 378), L=L, max_order=2, frontend='numpy')
print(np.shape(scattering))

rootdir = '/Users/Hurum/Documents/simulation_new'
zeroth_ord=[]
first_ord_1= []
first_ord_2= []
second_ord=[]

for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        sub_d = os.path.join(d+'/lya_forest_spectra.hdf5')
        print(sub_d)


        lya_file = h5py.File(sub_d, 'r')
        all_tau = lya_file['tau']['H']['1']['1215'][:]
        all_flux = np.exp(-all_tau)
        boxsize = lya_file['Header'].attrs['box']/1000
        z= 4.2
        res_in_cmp = (100 /cosmo.H(z).value)*(1+z) * cosmo.h
        num_pix = res_in_cmp / (boxsize/all_flux.shape[1])
        coarse_flux = scipy.ndimage.gaussian_filter1d(all_flux , sigma=num_pix, axis=1, mode='wrap')
        all_flux= coarse_flux



        scat_coeffs= scattering(all_flux)
        scat_coeffs= -scat_coeffs
        print(np.shape(scat_coeffs))
        
        mean_0=[]
        lor=0
        for i in range(0,8000):
            lor= lor+ scat_coeffs[0,i,:]
            mean_0.append(scat_coeffs[0,i,:])
        print(np.shape(lor))
      
        zeroth_ord.append(lor/8000)
        
        mean_0=[]
        lor=0
        for i in range(0,8000):
            lor= lor+ scat_coeffs[1,i,:]
            mean_0.append(scat_coeffs[1,i,:])
        print(np.shape(lor))
      
        first_ord_1.append(lor/8000)
        
        
        mean_0=[]
        lor=0
        for i in range(0,8000):
            lor= lor+ scat_coeffs[2,i,:]
            mean_0.append(scat_coeffs[2,i,:])
        print(np.shape(lor))
      
        first_ord_2.append(lor/8000)
        
        
        mean_0=[]
        lor=0
        for i in range(0,8000):
            lor= lor+ scat_coeffs[3,i,:]
            mean_0.append(scat_coeffs[3,i,:])
        print(np.shape(lor))
      
        second_ord.append(lor/8000)
       

plt.rcParams["figure.figsize"] = [29,7]
plt.rc("axes", linewidth=2.0)
plt.rc("lines", markeredgewidth=3)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)

fig, axs = plt.subplots(nrows=1, ncols=3)

plt.subplots_adjust(wspace=0.4, hspace=0.4)

axs[0].plot(zeroth_ord[0], label='ns0.897As1.55e-09heat_slope-0.3heat_amp0.9', linewidth=4, linestyle='solid', color='blue')
axs[0].plot(zeroth_ord[1], label='ns0.897As1.9e-09heat_slope-0.5heat_amp0.9', linewidth=4, linestyle='dashed', color='orange')
axs[0].plot(zeroth_ord[2], label='ns0.897As1.9e-09heat_slope-0.5heat_amp0.9', linewidth=4, linestyle='dashdot', color='pink')
axs[0].plot(zeroth_ord[3], label='ns0.897As1.9e-09heat_slope-0.1heat_amp0.9', linewidth=4, linestyle='dotted',  color='green')
axs[0].plot(zeroth_ord[4], label='ns0.897As1.9e-09heat_slope-0.3heat_amp1.4', linewidth=4, linestyle=(0, (3, 1)), color='purple')
axs[0].plot(zeroth_ord[5], label='ns0.897As1.9e-09heat_slope-0.3heat_amp0.65', linewidth=4, linestyle=(0, (1, 1)), color='red')
axs[0].plot(zeroth_ord[6], label='ns0.897As1.9e-09heat_slope-0.7heat_amp0.9', linewidth=4, linestyle='solid', dashes=[5, 2], color='teal')
axs[0].plot(zeroth_ord[7], label='ns0.897As2.25e-09heat_slope-0.3heat_amp0.9', linewidth=4, linestyle='solid', dashes=[2, 2, 10, 2], color='olive')
axs[0].plot(zeroth_ord[8], label='ns0.897As1.2e-09heat_slope-0.3heat_amp0.9', linewidth=4, linestyle='solid', dashes=[5, 2, 1, 2], color='brown')
axs[0].plot(zeroth_ord[9], label='ns0.897As1.9e-09heat_slope-0.3heat_amp0.9', linewidth=4, linestyle='solid', dashes=[2, 2, 2, 2, 10, 2], color='magenta')
axs[0].plot(zeroth_ord[10], label='ns0.8As1.9e-09heat_slope-0.3heat_amp0.9', linewidth=4, linestyle='solid', dashes=[10, 5, 2, 5], color='gray')



axs[1].plot(first_ord_1[0], label= 'ns0.897As1.55e-09heat_slope-0.3heat_amp0.9', linewidth=4, linestyle='solid', color='blue')
axs[1].plot(first_ord_1[1], label= 'ns0.897As1.9e-09heat_slope-0.5heat_amp0.9', linewidth =4, linestyle='dashed', color='orange')
axs[1].plot(first_ord_1[2], label= 'ns0.897As1.9e-09heat_slope-0.5heat_amp0.9', linewidth =4, linestyle='dashdot', color='pink')
axs[1].plot(first_ord_1[3], label= 'ns0.897As1.9e-09heat_slope-0.1heat_amp0.9', linewidth =4, linestyle='dotted',  color='green')
axs[1].plot(first_ord_1[4], label= 'ns0.897As1.9e-09heat_slope-0.3heat_amp1.4', linewidth =4, linestyle=(0, (3, 1)), color='purple')
axs[1].plot(first_ord_1[5], label= 'ns0.897As1.9e-09heat_slope-0.3heat_amp0.65', linewidth =4, linestyle=(0, (1, 1)), color='red')
axs[1].plot(first_ord_1[6], label= 'ns0.897As1.9e-09heat_slope-0.7heat_amp0.9', linewidth =4, linestyle='solid', dashes=[5, 2], color='teal')
axs[1].plot(first_ord_1[7], label= 'ns0.897As2.25e-09heat_slope-0.3heat_amp0.9', linewidth =4, linestyle='solid', dashes=[2, 2, 10, 2], color='olive')
axs[1].plot(first_ord_1[8], label= 'ns0.897As1.2e-09heat_slope-0.3heat_amp0.9', linewidth =4, linestyle='solid', dashes=[5, 2, 1, 2], color='brown')
axs[1].plot(first_ord_1[9], label= 'ns0.897As1.9e-09heat_slope-0.3heat_amp0.9', linewidth =4, linestyle='solid', dashes=[2, 2, 2, 2, 10, 2], color='magenta')
axs[1].plot(first_ord_1[10], label= 'ns0.8As1.9e-09heat_slope-0.3heat_amp0.9', linewidth =4, linestyle='solid', dashes=[10, 5, 2, 5], color='gray')


axs[2].plot(second_ord[0], label= 'ns0.897As1.55e-09heat_slope-0.3heat_amp0.9', linewidth=4, linestyle='solid', color='blue')
axs[2].plot(second_ord[1], label= 'ns0.897As1.9e-09heat_slope-0.5heat_amp0.9', linewidth =4, linestyle='dashed', color='orange')
axs[2].plot(second_ord[2], label= 'ns0.897As1.9e-09heat_slope-0.5heat_amp0.9', linewidth =4, linestyle='dashdot', color='pink')
axs[2].plot(second_ord[3], label= 'ns0.897As1.9e-09heat_slope-0.1heat_amp0.9', linewidth =4, linestyle='dotted',  color='green')
axs[2].plot(second_ord[4], label= 'ns0.897As1.9e-09heat_slope-0.3heat_amp1.4', linewidth =4, linestyle=(0, (3, 1)), color='purple')
axs[2].plot(second_ord[5], label= 'ns0.897As1.9e-09heat_slope-0.3heat_amp0.65', linewidth =4, linestyle=(0, (1, 1)), color='red')
axs[2].plot(second_ord[6], label= 'ns0.897As1.9e-09heat_slope-0.7heat_amp0.9', linewidth =4, linestyle='solid', dashes=[5, 2], color='teal')
axs[2].plot(second_ord[7], label= 'ns0.897As2.25e-09heat_slope-0.3heat_amp0.9', linewidth =4, linestyle='solid', dashes=[2, 2, 10, 2], color='olive')
axs[2].plot(second_ord[8], label= 'ns0.897As1.2e-09heat_slope-0.3heat_amp0.9', linewidth =4, linestyle='solid', dashes=[5, 2, 1, 2], color='brown')
axs[2].plot(second_ord[9], label= 'ns0.897As1.9e-09heat_slope-0.3heat_amp0.9', linewidth =4, linestyle='solid', dashes=[2, 2, 2, 2, 10, 2], color='magenta')
axs[2].plot(second_ord[10], label= 'ns0.8As1.9e-09heat_slope-0.3heat_amp0.9', linewidth =4, linestyle='solid', dashes=[10, 5, 2, 5], color='gray')



axs[0].set_title('Zeroth Order', size = 25)
axs[1].set_title('First Order', size = 25)
axs[2].set_title('Second Order', size = 25)
    

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.3), fontsize = 25)

plt.savefig('cosmo_dep_coeffcients.pdf', dpi=300, bbox_inches='tight')
plt.show()

