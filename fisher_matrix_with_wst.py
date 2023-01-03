import os
import h5py
import numpy as np
import cv2
from kymatio import Scattering2D
from PIL import Image
import matplotlib.pyplot as plt
#from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from astropy.cosmology import Planck15 as cosmo
import scipy
from scipy.ndimage import gaussian_filter1d

try:
    xrange(1)
except NameError:
    xrange = range

    
    
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def _powerspectrum(inarray, axis=-1):
    """Compute the power spectrum of the input using np.fft"""
    rfftd = np.fft.rfft(inarray, axis=axis)
    # Want P(k)= F(k).re*F(k).re+F(k).im*F(k).im
    power = np.abs(rfftd)**2
    #Normalise the FFT so it is independent of input size.
    power /= np.shape(inarray)[axis]**2
    return power


def add_noise(snr, flux, spec_num=-1):
 
        nbins=flux.shape[1]
        noise_array = np.array([])
        if np.size(np.shape(flux)) == 1:
            lines = 1
        else:
            lines = np.shape(flux)[0]
        #This is to get around the type rules.
        if lines == 1:
            #This ensures that we always get the same noise for the same spectrum
            np.random.seed(spec_num)
            flux += np.random.normal(0, 1./snr[spec_num], nbins)
        else:
            for ii in xrange(lines):
                np.random.seed(ii)
                noise = np.random.normal(0, 1./snr[ii], nbins)
                noise_array = np.append(noise_array, noise)
                flux[ii]+= noise
        return (flux, noise_array)


def add_cont_error(CE, flux, spec_num=-1, u_delta=0.6, l_delta=-0.6):
        """Adding the Continuum error to spectra. If you want to add both random noise and continuum error, first add
        the continuum error and then the random noise. Follow the prescription in eq 2 of arxiv:2112.03930:
        F_obs = F_sim / (1 + delta) and delta = Gaussian(0, CE) where delta_L < delta < delta_u
        Parameters:
        CE : the stdev of the gaussian noise
        flux : an array of spectra (flux)  we want to add noise to
        spec_num : the index to spectra we want to add nose to. Leave it as -1 to add the noise to all spectra.
        u_delta, l_delta : upper and lower limit of the delta parameter
        """
        if np.size(np.shape(flux)) == 1:
            lines = 1
        else:
            lines = np.shape(flux)[0]
        #This is to get around the type rules
        if lines == 1:
            #This ensures that we always get the same noise for the same spectrum and is differen from seed for rand noise
            np.random.seed(2*spec_num)
            delta = np.random.normal(0, CE[spec_num])
            # Use lower and upper limit of delta from 2sigma for the highest CE in the survey
            while (delta < l_delta) or (delta > u_delta):
                delta = np.random.normal(0, CE[spec_num])
            flux /= (1.0 + delta)
        else:
            delta = np.empty(lines)
            for ii in xrange(lines):
                np.random.seed(2*ii)
                delta[ii] = np.random.normal(0, CE[ii])
                while (delta[ii] < l_delta) or (delta[ii] > u_delta):
                    delta[ii] = np.random.normal(0, CE[ii])
                flux[ii,:] /= (1.0 + delta[ii])
        return (flux , delta)
      
L = 6
J = 3
scattering = Scattering2D(J=J, shape=(200, 378), L=L, max_order=2, frontend='numpy')
print(np.shape(scattering))   



rootdir = '/Users/Hurum/Documents/simulation_new'
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        sub_d = os.path.join(d+'/lya_forest_spectra.hdf5')


        lya_file = h5py.File(sub_d, 'r')
        all_tau = lya_file['tau']['H']['1']['1215'][:]
        all_flux = np.exp(-all_tau[:200])
        boxsize = lya_file['Header'].attrs['box']/1000
        z= 1.92
        res_in_cmp = (100 /cosmo.H(z).value)*(1+z) * cosmo.h
        num_pix = res_in_cmp / (boxsize/all_flux.shape[1])
        coarse_flux = scipy.ndimage.gaussian_filter1d(all_flux , sigma=num_pix, axis=1, mode='wrap')
        all_flux= coarse_flux

        #plt.imshow(all_flux[:400]) #400,378
        #plt.savefig(d+'.jpeg')


        scat_coeffs= scattering(all_flux)
        scat_coeffs= -scat_coeffs

        for i in range(0, len(scat_coeffs)):
            np.savetxt(d+ str(i)+'.dat', scat_coeffs[i])
            
rootdir_ = '/Users/Hurum/Documents/simulation_new'
sim_list=['ns0.8As1.9e-09heat_slope-0.3heat_amp0.9hub0.7',
'ns0.849As1.9e-09heat_slope-0.3heat_amp0.9hub0.7',
'ns0.897As1.9e-09heat_slope-0.3heat_amp0.9hub0.7',
'ns0.897As1.9e-09heat_slope-0.5heat_amp0.9hub0.7',
'ns0.897As1.9e-09heat_slope-0.1heat_amp0.9hub0.7',
'ns0.897As1.9e-09heat_slope-0.7heat_amp0.9hub0.7',
'ns0.897As1.9e-09heat_slope-0.3heat_amp1.4hub0.7',
'ns0.897As1.9e-09heat_slope-0.3heat_amp0.65hub0.7',
'ns0.897As1.9e-09heat_slope0.1heat_amp0.9hub0.7', ##BOTH heat slope and amp different
'ns0.897As2.25e-09heat_slope-0.3heat_amp0.9hub0.7',
'ns0.897As1.2e-09heat_slope-0.3heat_amp0.9hub0.7',
'ns0.897As1.55e-09heat_slope-0.3heat_amp0.9hub0.7 ']

ns=[]
As=[]
slope=[]
amp=[]

dS_1 = []
dS_2 = []
dS_3 = []

dS_4 = []
dS_5 = []
dS_6 = []

dS_7 = []
dS_8 = []
dS_9 = []

dS_10 = []
dS_11 = []
dS_12 = []


for i in range(0,127):
    img = str(i)+'.dat'

    coeff_sim_1= np.loadtxt(sim_list[0]+img)
    coeff_sim_2= np.loadtxt(sim_list[1]+img)
    coeff_sim_3= np.loadtxt(sim_list[2]+img)
    coeff_sim_4= np.loadtxt(sim_list[3]+img)
    coeff_sim_5= np.loadtxt(sim_list[4]+img)
    coeff_sim_6= np.loadtxt(sim_list[5]+img)
    coeff_sim_7= np.loadtxt(sim_list[6]+img)
    coeff_sim_8= np.loadtxt(sim_list[7]+img)
    coeff_sim_9= np.loadtxt(sim_list[8]+img)
    coeff_sim_10= np.loadtxt(sim_list[9]+img)
    coeff_sim_11= np.loadtxt(sim_list[10]+img)
    coeff_sim_12= np.loadtxt(sim_list[11]+img)



    ns_dp = [(.8-.849)/.897, (.849-.897)/.897, (.8- .897)/.897]
    heatslope_dp = [(.5-.1)/.3 ,(.1-.7)/.3, (.5-.7)/.3]
    amp_dp = [(1.4-.65)/.9]
    As_dp = [(2.25e-9-1.2e-9)/(1.9e-9), (1.2e-9-1.55e-9)/(1.9e-9),(1.55e-9-2.25e-9)/(1.9e-9)]

    dp = [ns_dp, heatslope_dp, amp_dp, As_dp]


    dS_1.append(rmse(coeff_sim_1,coeff_sim_2)/ns_dp[0])
    dS_2.append(rmse(coeff_sim_2, coeff_sim_3)/ns_dp[1])
    dS_3.append(rmse(coeff_sim_1, coeff_sim_3)/ns_dp[2])

    dS_4.append(rmse(coeff_sim_4,coeff_sim_5)/heatslope_dp[0])
    dS_5.append(rmse(coeff_sim_5, coeff_sim_6)/heatslope_dp[1])
    dS_6.append(rmse(coeff_sim_4, coeff_sim_6)/heatslope_dp[2])

    dS_7.append(rmse(coeff_sim_7,coeff_sim_8)/amp_dp[0])
    #dS_8.append(rmse(coeff_sim_8, coeff_sim_9)/amp_dp[1])
    #dS_9.append(rmse(coeff_sim_7, coeff_sim_9)/amp_dp[2])

    dS_10.append(rmse(coeff_sim_10,coeff_sim_11)/As_dp[0])
    dS_11.append(rmse(coeff_sim_11, coeff_sim_12)/As_dp[1])
    dS_12.append(rmse(coeff_sim_10, coeff_sim_12)/As_dp[2])
    
ds_dp =[(np.sum(dS_1)+np.sum(dS_2)+np.sum(dS_3))/3, (np.sum(dS_4)+np.sum(dS_5)+np.sum(dS_6))/3,np.sum(dS_7),(np.sum(dS_10)+np.sum(dS_11)+np.sum(dS_12))/3]
ds_dp_first_ord =[(np.sum(dS_1[:9])+np.sum(dS_2[:9])+np.sum(dS_3[:9]))/3, (np.sum(dS_4[:9])+np.sum(dS_5[:9])+np.sum(dS_6[:9]))/3,np.sum(dS_7[:9]),(np.sum(dS_10[:9])+np.sum(dS_11[:9])+np.sum(dS_12[:9]))/3]
ds_dp_second_ord =[(np.sum(dS_1[-108:])+np.sum(dS_2[-108:])+np.sum(dS_3[-108:]))/3, (np.sum(dS_4[-108:])+np.sum(dS_5[-108:])+np.sum(dS_6[-108:]))/3,np.sum(dS_7[-108:]),(np.sum(dS_10[-108:])+np.sum(dS_11[-108:])+np.sum(dS_12[-108:]))/3]


lya_file = h5py.File('/Users/Hurum/Documents/simulation_new/ns0.897As1.9e-09heat_slope-0.3heat_amp0.9hub0.7/lya_forest_spectra.hdf5', 'r')
all_tau = lya_file['tau']['H']['1']['1215'][:]
all_flux = np.exp(-all_tau[:200])
boxsize = lya_file['Header'].attrs['box']/1000
z= 1.92
res_in_cmp = (100 /cosmo.H(z).value)*(1+z) * cosmo.h
num_pix = res_in_cmp / (boxsize/all_flux.shape[1])
coarse_flux = scipy.ndimage.gaussian_filter1d(all_flux , sigma=num_pix, axis=1, mode='wrap')
all_flux= coarse_flux


st= scattering(all_flux)
st= -st

CNR = 2*np.exp(np.random.normal(0.5323449534337695, 0.36223889354351285, size=all_flux.shape[0]))
CE = 0.24*CNR**(0.86)
noisy_flux,delta= add_cont_error(CE, all_flux, spec_num=-1, u_delta=0.6, l_delta=-0.6)
noisy_flux, dn = add_noise(CNR, noisy_flux)

scat_coeffs_noise= scattering(noisy_flux)
scat_coeffs_noise= -scat_coeffs_noise

diff=[]
diff_first_order=[]
diff_second_order=[]
for i in range(0,127):
    diff.append(rmse(scat_coeffs_noise[i], st[i]))
for i in range(0,18):
    diff_first_order.append(rmse(scat_coeffs_noise[i], st[i]))
for i in range(19,127):
    diff_second_order.append(rmse(scat_coeffs_noise[i], st[i]))
    
diff= np.sqrt(np.sum(diff))
diff_first_order= np.sqrt(np.sum(diff_first_order))
diff_second_order= np.sqrt(np.sum(diff_second_order))


dn_ds= (np.mean(dn)/diff)
dn_ds_first_order= (np.mean(dn)/diff_first_order)
dn_ds_second_order= (np.mean(dn)/diff_second_order)


F= [ds_dp[0]**2*dn_ds**2,ds_dp[1]**2*dn_ds**2,ds_dp[2]**2*dn_ds**2,ds_dp[3]**2*dn_ds**2]
F_first_order= [ds_dp_first_ord[0]**2*dn_ds_first_order**2,ds_dp_first_ord[1]**2*dn_ds_first_order**2,ds_dp_first_ord[2]**2*dn_ds_first_order**2,ds_dp_first_ord[3]**2*dn_ds_first_order**2]
F_second_order= [ds_dp_second_ord[0]**2*dn_ds_second_order**2,ds_dp_second_ord[1]**2*dn_ds_second_order**2,ds_dp_second_ord[2]**2*dn_ds_second_order**2,ds_dp_second_ord[3]**2*dn_ds_second_order**2]



