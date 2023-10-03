import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
import scipy
from scipy.ndimage import gaussian_filter1d
from kymatio.numpy import Scattering1D
from fake_spectra import fluxstatistics as fs
#from fake_spectra import spectra as spec
from scipy.ndimage import gaussian_filter1d as gf

try:
    xrange(1)
except NameError:
    xrange = range

def truncated_gaussian_samples(mean, std_dev, a, b, num_samples_):
    samples = []
    while len(samples) < num_samples_:
        value = np.random.normal(mean, std_dev)
        if a <= value <= b:
            samples.append(value)
    return samples


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
                noise = truncated_gaussian_samples(0, 1./snr[ii], -np.min(flux[ii]), np.max(flux[ii]), nbins)
                noise_array = np.append(noise_array, noise)

                flux[ii]+= noise

        return (flux, noise_array)


def get_mean_flux(z, metal=False) :
    """ get the mean flux used in LATIS Faucher-Giguere 2008"""
    if metal :
        # The below is not good for HI absorption as includes the effect of metals
        return np.exp(-0.001845*(1+z)**3.924)
    else :
        # The below is good for only HI absorptions, does not include metal absorption
        return np.exp(-0.001330*(1+z)**4.094)

def correct_mean_flux(tau, mean_flux_desired, ind=None):
    """ returns the corrected flux to have a desired mean flux
    arguments:
    tau : optical depth BEFORE adding noise to it
    ind: indices to spectra with NHI < 10^19. Only them being used for finding the scale. This is according to 
    Faucher-Giguere et al 2008. mean flux without including any metals. 
    mean_flux_desired:
    returns: The flux after scaling the optical depth to have mean_flux_desired = <e^(-scale * tau)>

    """
    if ind is not None:
        scale = fstat.mean_flux(tau[ind], mean_flux_desired)
    else :
        scale = fstat.mean_flux(tau, mean_flux_desired)
        
    flux = np.exp(-scale * tau)

def scale_factor_to_redshift(scale_factor):
    return (1 / scale_factor) - 1



sim_res= 100
z= scale_factor_to_redshift(.29)

J= 5
T= 370
Q= 1
scattering = Scattering1D(J, T, Q)


st_coeff= []

ps_mean=[]
lya_file = h5py.File('/rhome/htohf001/bigdata/z=2/ns0.897As1.9e-09heat_slope-0.3heat_amp0.9hub0.7/lya_forest_spectra.hdf5', 'r')
all_tau = lya_file['tau']['H']['1']['1215'][:]
scale  = get_mean_flux(z=z) 
all_flux= np.exp(-all_tau*scale)

for i in range (0,10):
	CNR = np.exp(np.random.normal(0.532, 0.362, 32000))
	noisy_flux, dn = add_noise(CNR, all_flux)
	boxsize = lya_file['Header'].attrs['box']/1000
	z= scale_factor_to_redshift(.29)
	res_in_cmp = (sim_res /cosmo.H(z).value)*(1+z) * cosmo.h
	num_pix = res_in_cmp / (boxsize/all_flux.shape[1])
	coarse_flux = scipy.ndimage.gaussian_filter1d(noisy_flux , sigma=num_pix, axis=1, mode='wrap')
	all_flux= coarse_flux

	tau_ = -np.log(all_flux)
	vmax = 4177
	spec_res = vmax/tau_.shape[1]
	kf, mean_flux_power = fs.flux_power(tau_, vmax= vmax, spec_res= spec_res)
	mean_flux_power = mean_flux_power*kf/np.pi
	ps=[]
	for i in range(0,len(kf)):
		if kf[i]< 0.5:
			ps.append(mean_flux_power[i])

	ps_mean.append(np.mean(ps))
	noisy_flux = all_flux/np.mean(all_flux) -1

	coeff_1d=[]
	for i in range(0,32000):
		coeffs = scattering(noisy_flux[i])
		coeffs = coeffs.mean(axis=1)
		coeff_1d.append(coeffs)
	coeff_1d= np.array(coeff_1d)
	st_coeff.append(coeff_1d.mean(axis=0))
    



pid = os.getpid()
np.savetxt(f'stz2_J5Q1_8-23{pid}.txt',st_coeff)
np.savetxt(f'psz2_8-23{pid}.txt',ps_mean)
