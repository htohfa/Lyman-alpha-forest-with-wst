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


L = 6  # Set the size of the scattering grid to 6
J = 3  # Set the number of wavelet scales to 3
scattering = Scattering2D(J=J, shape=(200, 378), L=L, max_order=2, frontend='numpy')  # Create a 2D scattering object with the specified parameters and using the numpy frontend
print(np.shape(scattering))  # Print the shape of the scattering object as a tuple.




rootdir = '/Users/Hurum/Documents/simulation_new'


# Loop over all files in the directory and its subdirectories
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        sub_d = os.path.join(d+'/lya_forest_spectra.hdf5')
        print(sub_d)

        # Open the HDF5 file containing the Lyman-alpha forest spectra
        lya_file = h5py.File(sub_d, 'r')

        # Extract the optical depth data
        all_tau = lya_file['tau']['H']['1']['1215'][:]

        # Get the rescaling factor for the data based on the observed mean optical depth at the given redshift
        rescale_factor = get_mean_flux_scale(all_tau, obs_mean_tau(scale_factor_to_redshift(.192)), all_tau.shape[1], 10**(-10))

        # Rescale the optical depth data and convert to flux
        rescaled_tau = rescale_factor*all_tau[:200]
        all_flux = np.exp(-rescaled_tau)

        # Fix the pixel size of the data based on the simulation resolution and box size
        boxsize = lya_file['Header'].attrs['box']/1000
        z= scale_factor_to_redshift(.192)
        res_in_cmp = (sim_res /cosmo.H(z).value)*(1+z) * cosmo.h
        num_pix = res_in_cmp / (boxsize/all_flux.shape[1])
        coarse_flux = scipy.ndimage.gaussian_filter1d(all_flux , sigma=num_pix, axis=1, mode='wrap')
        all_flux= coarse_flux

        # Calculate the scattering coefficients of the flux data
        scat_coeffs= scattering(all_flux)
        scat_coeffs= -scat_coeffs

        # Save the scattering coefficients for each order
        for i in range(0, len(scat_coeffs)):
            np.savetxt(d+ str(i)+'.dat', scat_coeffs[i])


rootdir_ = '/Users/Hurum/Documents/simulation_new'

# list of simulation strings to load data from
sim_list=['ns0.8As1.9e-09heat_slope-0.3heat_amp0.9hub0.7','ns0.849As1.9e-09heat_slope-0.3heat_amp0.9hub0.7','ns0.897As1.9e-09heat_slope-0.3heat_amp0.9hub0.7','ns0.897As1.9e-09heat_slope-0.5heat_amp0.9hub0.7','ns0.897As1.9e-09heat_slope-0.1heat_amp0.9hub0.7','ns0.897As1.9e-09heat_slope-0.7heat_amp0.9hub0.7','ns0.897As1.9e-09heat_slope-0.3heat_amp1.4hub0.7','ns0.897As1.9e-09heat_slope-0.3heat_amp0.65hub0.7','ns0.897As1.9e-09heat_slope0.1heat_amp0.9hub0.7', ##BOTH heat slope and amp different'ns0.897As2.25e-09heat_slope-0.3heat_amp0.9hub0.7','ns0.897As1.2e-09heat_slope-0.3heat_amp0.9hub0.7','ns0.897As1.55e-09heat_slope-0.3heat_amp0.9hub0.7 ']

# initialize lists to store values of various parameters
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

# loop through each image and calculate the change in each parameter
for i in range(0,127):
    img = str(i)+'.dat'

    # load simulation data for each simulation string and image
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

    # calculate the change in ns, heatslope, amp, and As for each simulation
    ns_dp = [(.8-.849)/.897, (.849-.897)/.897, (.8- .897)/.897]
    heatslope_dp = [(.5-.1)/.3 ,(.1-.7)/.3, (.5-.7)/.3]
    amp_dp = [(1.4-.65)/.9]
    As_dp = [(2.25e-9-1.2e-9)/(1.9e-9), (1.2e-9-1.55e-9)/(1.9e-9),(1.55e-9-2.25e-9)/(1.9e-9)]

    dp = [ns_dp, heatslope_dp, amp_dp, As_dp]

    #calculating ds/dp


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


#first order, second order, first and second order
ds_dp =[(np.sum(dS_1)+np.sum(dS_2)+np.sum(dS_3))/3, (np.sum(dS_4)+np.sum(dS_5)+np.sum(dS_6))/3,np.sum(dS_7),(np.sum(dS_10)+np.sum(dS_11)+np.sum(dS_12))/3]
ds_dp_first_ord =[(np.sum(dS_1[:9])+np.sum(dS_2[:9])+np.sum(dS_3[:9]))/3, (np.sum(dS_4[:9])+np.sum(dS_5[:9])+np.sum(dS_6[:9]))/3,np.sum(dS_7[:9]),(np.sum(dS_10[:9])+np.sum(dS_11[:9])+np.sum(dS_12[:9]))/3]
ds_dp_second_ord =[(np.sum(dS_1[-108:])+np.sum(dS_2[-108:])+np.sum(dS_3[-108:]))/3, (np.sum(dS_4[-108:])+np.sum(dS_5[-108:])+np.sum(dS_6[-108:]))/3,np.sum(dS_7[-108:]),(np.sum(dS_10[-108:])+np.sum(dS_11[-108:])+np.sum(dS_12[-108:]))/3]

# calculating first and second order scattering coefficients for the input data

ds_dp =[(np.sum(dS_1)+np.sum(dS_2)+np.sum(dS_3))/3,         (np.sum(dS_4)+np.sum(dS_5)+np.sum(dS_6))/3,         np.sum(dS_7),        (np.sum(dS_10)+np.sum(dS_11)+np.sum(dS_12))/3]
ds_dp_first_ord =[(np.sum(dS_1[:9])+np.sum(dS_2[:9])+np.sum(dS_3[:9]))/3,
                  (np.sum(dS_4[:9])+np.sum(dS_5[:9])+np.sum(dS_6[:9]))/3,
                  np.sum(dS_7[:9]),
                  (np.sum(dS_10[:9])+np.sum(dS_11[:9])+np.sum(dS_12[:9]))/3]
ds_dp_second_ord =[(np.sum(dS_1[-108:])+np.sum(dS_2[-108:])+np.sum(dS_3[-108:]))/3,
                   (np.sum(dS_4[-108:])+np.sum(dS_5[-108:])+np.sum(dS_6[-108:]))/3,
                   np.sum(dS_7[-108:]),
                   (np.sum(dS_10[-108:])+np.sum(dS_11[-108:])+np.sum(dS_12[-108:]))/3]

# reading in a central simulation spectra file and preparing the data for the scattering calculation
lya_file = h5py.File('/Users/Hurum/Documents/simulation_new/ns0.897As1.9e-09heat_slope-0.3heat_amp0.9hub0.7/lya_forest_spectra.hdf5', 'r')
all_tau = lya_file['tau']['H']['1']['1215'][:]
rescale_factor = get_mean_flux_scale(all_tau, obs_mean_tau(scale_factor_to_redshift(.192)), all_tau.shape[1], 10**(-10))
rescaled_tau = rescale_factor * all_tau[:200]
all_flux = np.exp(-rescaled_tau)

boxsize = lya_file['Header'].attrs['box'] / 1000
z = scale_factor_to_redshift(.192)
res_in_cmp = (sim_res / cosmo.H(z).value) * (1 + z) * cosmo.h
num_pix = res_in_cmp / (boxsize / all_flux.shape[1])

# smoothing the data using a Gaussian filter
coarse_flux = scipy.ndimage.gaussian_filter1d(all_flux, sigma=num_pix, axis=1, mode='wrap')
all_flux = coarse_flux

# calculating the scattering coefficients for the smoothed data
st = scattering(all_flux)
st = -st

# adding noise to the data
CNR = noise_amp * np.exp(np.random.normal(0.5323449534337695, 0.36223889354351285, size=all_flux.shape[0]))
CE = 0.24 * CNR**(0.86)
noisy_flux,delta= add_cont_error(CE, all_flux, spec_num=-1, u_delta=0.6, l_delta=-0.6)
noisy_flux, dn = add_noise(CNR, noisy_flux)

#calculating scatter coefficient of the central simulation
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

# Calculate the root mean squared error (RMSE) for the entire range, first-order and second-order ranges of scat_coeffs_noise and st
diff= np.sqrt(np.sum(diff))
diff_first_order= np.sqrt(np.sum(diff_first_order))
diff_second_order= np.sqrt(np.sum(diff_second_order))

# Calculate the derivatives of the noise component of the data with respect to the four parameters: ds_dp[0], ds_dp[1], ds_dp[2], ds_dp[3] for the entire range, first-order and second-order ranges.
dn_ds= (np.mean(dn)/diff)
dn_ds_first_order= (np.mean(dn)/diff_first_order)
dn_ds_second_order= (np.mean(dn)/diff_second_order)

# Calculate the Fisher matrix (F) for the entire range, first-order and second-order ranges.
F= [ds_dp[0]**2*dn_ds**2,ds_dp[1]**2*dn_ds**2,ds_dp[2]**2*dn_ds**2,ds_dp[3]**2*dn_ds**2]
F_first_order= [ds_dp_first_ord[0]**2*dn_ds_first_order**2,ds_dp_first_ord[1]**2*dn_ds_first_order**2,ds_dp_first_ord[2]**2*dn_ds_first_order**2,ds_dp_first_ord[3]**2*dn_ds_first_order**2]
F_second_order= [ds_dp_second_ord[0]**2*dn_ds_second_order**2,ds_dp_second_ord[1]**2*dn_ds_second_order**2,ds_dp_second_ord[2]**2*dn_ds_second_order**2,ds_dp_second_ord[3]**2*dn_ds_second_order**2]



# Save the Fisher matrix (F) to a text file
np.savetxt('st_1_130.txt',F)
