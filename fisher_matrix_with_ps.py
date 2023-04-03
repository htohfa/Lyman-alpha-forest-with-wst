import os
import h5py
import numpy as np
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

def get_mean_flux_scale(tau, desired_mean_flux, nbins, tol):
    """"Rescales the optical depth values by adjusting the scaling factor scale
     until the mean flux of the spectrum matches a desired value desired_mean_flux within a tolerance tol.
      The function calculates the exponential factor for each bin in the spectrum and uses it to calculate the mean
      flux and the tau-weighted mean flux."""
    newscale =1  # Initialize the newscale to 1
    scale=0  # Initialize the scale to 0
    while np.abs(newscale-scale) > (tol*newscale):  # Run this loop until the change in scale is within a certain tolerance level.

        scale= newscale  # Set the current scale to the new scale
        mean_flux =0  # Initialize the mean flux to 0
        tau_mean_flux =0  # Initialize the tau-weighted mean flux to 0
        nbins_used=0  # Initialize the number of bins used to 0
        tau_mean=[]
        for i in range(0, 32000):
            tau_mean.append(np.mean(tau[i]))  # Calculate the mean of each row in the input array tau and append to tau_mean list

        for i in range(0, nbins):
            temp= np.exp(-scale*tau_mean[i])  # Calculate the exponential factor for this bin
            mean_flux= mean_flux+temp  # Add the exponential factor to the mean flux
            tau_mean_flux = tau_mean_flux+temp*tau_mean[i]  # Add the exponential factor times tau_mean to the tau-weighted mean flux
            nbins_used=nbins_used+1  # Increment the number of bins used

        newscale=scale+(mean_flux- desired_mean_flux * nbins_used)/tau_mean_flux  # Calculate the new scale using the current scale, desired mean flux, and tau-weighted mean flux

        if newscale <= 0:
            newscale=1e-10  # If the new scale is less than or equal to 0, set it to a small value to avoid division by 0

    return newscale  # Return the final scale value


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())  # Return the root-mean-squared error between the predicted and target values

def _powerspectrum(inarray, axis=-1):
    """Compute the power spectrum of the input using np.fft"""
    rfftd = np.fft.rfft(inarray, axis=axis)
    # Want P(k)= F(k).re*F(k).re+F(k).im*F(k).im
    power = np.abs(rfftd)**2
    #Normalise the FFT so it is independent of input size.
    power /= np.shape(inarray)[axis]**2
    return power

    
def add_noise(snr, flux, spec_num=-1):
        """Compute a Gaussian noise vector from the flux variance and the SNR, as computed from optical depth
        Parameters:
        snr : an array of signal to noise ratio (constant along each sightine)
        flux : an array of spectra (flux)  we want to add noise to
        spec_num : the index to spectra we want to add nose to. Leave it as -1 to add the noise to all spectra.
        """
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


def obs_mean_tau(redshift):
    return 0.0023*(1.0+redshift)**3.65  # Calculate the observed mean optical depth given a redshift value


def scale_factor_to_redshift(scale_factor):
    return (1 / scale_factor) - 1  # Convert a scale factor to a corresponding redshift value


noise_amp= 1  # Set the amplitude of the noise to 1
sim_res= 100  # Set the resolution of the simulation to 100

#set the directory where the spectra are located
rootdir = '/Users/Hurum/Documents/simulation_new'

#loop through all files in the root directory
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        sub_d = os.path.join(d+'/lya_forest_spectra.hdf5')
        print(sub_d)


        lya_file = h5py.File(sub_d, 'r')
        # Extract the absorption of neutral hydrogen atoms at 1215 Angstroms
        all_tau = lya_file['tau']['H']['1']['1215'][:]
 
        # Calculate the rescaling factor based on the mean flux and observed mean tau
        rescale_factor = get_mean_flux_scale(all_tau, obs_mean_tau( scale_factor_to_redshift(.192)),all_tau.shape[1], 10**(-10))
        
        # Rescale the absorption
        rescaled_tau = rescale_factor*all_tau[:200]
        # Convert the rescaled absorption to flux
        all_flux = np.exp(-rescaled_tau)
        
        #adjusting the resolution
        boxsize = lya_file['Header'].attrs['box']/1000
        z= scale_factor_to_redshift(.192) #calculating the redshift
        res_in_cmp = (sim_res /cosmo.H(z).value)*(1+z) * cosmo.h
        num_pix = res_in_cmp / (boxsize/all_flux.shape[1])
        coarse_flux = scipy.ndimage.gaussian_filter1d(all_flux , sigma=num_pix, axis=1, mode='wrap')
        all_flux= coarse_flux
        
        # Calculate the power spectrum of the flux
        ps_= _powerspectrum(all_flux, axis=-1)
        np.savetxt(d+".dat", ps_)




rootdir_ = '/Users/Hurum/Documents/simulation_new'
sim_list=['ns0.8As1.9e-09heat_slope-0.3heat_amp0.9hub0.7',
'ns0.849As1.9e-09heat_slope-0.3heat_amp0.9hub0.7',
'ns0.897As1.9e-09heat_slope-0.3heat_amp0.9hub0.7',
'ns0.897As1.9e-09heat_slope-0.5heat_amp0.9hub0.7',
'ns0.897As1.9e-09heat_slope-0.1heat_amp0.9hub0.7',
'ns0.897As1.9e-09heat_slope-0.7heat_amp0.9hub0.7',
'ns0.897As1.9e-09heat_slope-0.3heat_amp1.4hub0.7',
'ns0.897As1.9e-09heat_slope-0.3heat_amp0.65hub0.7',
'ns0.897As1.9e-09heat_slope0.1heat_amp0.9hub0.7',  ##heat slope AND heat amp different so can't use 
'ns0.897As2.25e-09heat_slope-0.3heat_amp0.9hub0.7',
'ns0.897As1.2e-09heat_slope-0.3heat_amp0.9hub0.7','ns0.897As1.55e-09heat_slope-0.3heat_amp0.9hub0.7 ']

# initialize lists to store values of various parameters
ns=[]
As=[]
slope=[]
amp=[]

ps_1= np.loadtxt(sim_list[0]+".dat")
ps_2= np.loadtxt(sim_list[1]+".dat")
ps_3= np.loadtxt(sim_list[2]+".dat")
ps_4= np.loadtxt(sim_list[3]+".dat")
ps_5= np.loadtxt(sim_list[4]+".dat")
ps_6= np.loadtxt(sim_list[5]+".dat")
ps_7= np.loadtxt(sim_list[6]+".dat")
ps_8= np.loadtxt(sim_list[7]+".dat")
ps_9= np.loadtxt(sim_list[8]+".dat")
ps_10= np.loadtxt(sim_list[9]+".dat")
ps_11= np.loadtxt(sim_list[10]+".dat")
ps_12= np.loadtxt(sim_list[11]+".dat")


# calculate the change in ns, heatslope, amp, and As for each simulation
ns_dp = [(.8-.849)/.897, (.849-.897)/.897, (.8- .897)/.897]
heatslope_dp = [(.5-.1)/.3 ,(.1-.7)/.3, (.5-.7)/.3]
amp_dp = [(1.4-.65)/.9]
As_dp = [(2.25e-9-1.2e-9)/(1.9e-9), (1.2e-9-1.55e-9)/(1.9e-9),(1.55e-9-2.25e-9)/(1.9e-9)]

dp = [ns_dp, heatslope_dp, amp_dp, As_dp]

#calculating derivative of power spectra with change in parameters

dS_1 = rmse(ps_1,ps_2)
dS_2 = rmse(ps_2, ps_3)
dS_3 = rmse(ps_1, ps_3)

dS_4 = rmse(ps_4,ps_5)
dS_5 = rmse(ps_5, ps_6)
dS_6 = rmse(ps_4, ps_6)

dS_7 = rmse(ps_7,ps_8)
#dS_8 = rmse(ps_8, ps_9)
#dS_9 = rmse(ps_7, ps_9)

dS_10 = rmse(ps_10,ps_11)
dS_11 = rmse(ps_11, ps_12)
dS_12 = rmse(ps_10, ps_12)


ds_dp = [np.mean([dS_1/(dp[0][0]) ,dS_2/(dp[0][1]), dS_3/(dp[0][2])]),
                np.mean([dS_4/(dp[1][0]) ,dS_5/(dp[1][1]), dS_6/(dp[1][2])]),
                np.mean([dS_7/(dp[2][0])]),
                np.mean([dS_10/(dp[3][0]) ,dS_11/(dp[3][1]), dS_12/(dp[3][2])])]




power_spec= [ds_dp[0],ds_dp[1],ds_dp[2],ds_dp[3]]
np.savetxt('ps_.dat',power_spec)


#adjusting the central simulation to compare with

lya_file = h5py.File('/Users/Hurum/Documents/simulation_new/ns0.897As1.9e-09heat_slope-0.3heat_amp0.9hub0.7/lya_forest_spectra.hdf5', 'r')
all_tau = lya_file['tau']['H']['1']['1215'][:]

rescale_factor = get_mean_flux_scale(all_tau, obs_mean_tau( scale_factor_to_redshift(.192)),all_tau.shape[1], 10**(-10))
        
rescaled_tau = rescale_factor*all_tau[:200]
all_flux = np.exp(-rescaled_tau)



boxsize = lya_file['Header'].attrs['box']/1000
z=  scale_factor_to_redshift(.192)
res_in_cmp = (sim_res /cosmo.H(z).value)*(1+z) * cosmo.h
num_pix = res_in_cmp / (boxsize/all_flux.shape[1])
coarse_flux = scipy.ndimage.gaussian_filter1d(all_flux , sigma=num_pix, axis=1, mode='wrap')
all_flux= coarse_flux


ps_central = _powerspectrum(all_flux, axis=-1)

#adding noise
CNR = noise_amp*np.exp(np.random.normal(0.5323449534337695, 0.36223889354351285, size=all_flux.shape[0]))
CE = 0.24*CNR**(0.86)
noisy_flux,delta= add_cont_error(CE, all_flux, spec_num=-1, u_delta=0.6, l_delta=-0.6)
noisy_flux, dn = add_noise(CNR, noisy_flux)


ps_n= _powerspectrum(noisy_flux, axis=-1)


diff = rmse(ps_central,ps_n)
dn_ds= (np.mean(dn)/diff)

#calculating fisher matrix
F_ps= [power_spec[0]**2*dn_ds**2,power_spec[1]**2*dn_ds**2,power_spec[2]**2*dn_ds**2,power_spec[3]**2*dn_ds**2]



# Save the Fisher matrix (F_ps) to a text file
np.savetxt('ps_1_100.txt',F_ps)
