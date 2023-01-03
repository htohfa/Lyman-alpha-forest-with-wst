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
        #all_flux= all_flux[:378]

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
'ns0.897As1.2e-09heat_slope-0.3heat_amp0.9hub0.7',
'ns0.897As1.55e-09heat_slope-0.3heat_amp0.9hub0.7 ']

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



ns_dp = [(.8-.849)/.897, (.849-.897)/.897, (.8- .897)/.897]
heatslope_dp = [(.5-.1)/.3 ,(.1-.7)/.3, (.5-.7)/.3]
amp_dp = [(1.4-.65)/.9]
As_dp = [(2.25e-9-1.2e-9)/(1.9e-9), (1.2e-9-1.55e-9)/(1.9e-9),(1.55e-9-2.25e-9)/(1.9e-9)]

dp = [ns_dp, heatslope_dp, amp_dp, As_dp]


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

def gaussian_noise_added_sig(sig, snr):
  snr = (snr/100)*np.std(sig)
  noise = np.random.normal(0, snr, size = sig.shape)
  noisy_sig = sig + noise
  return noisy_sig,noise

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

lya_file = h5py.File('/Users/Hurum/Documents/simulation_new/ns0.897As1.9e-09heat_slope-0.3heat_amp0.9hub0.7/lya_forest_spectra.hdf5', 'r')
all_tau = lya_file['tau']['H']['1']['1215'][:]
all_flux = np.exp(-all_tau[:200])
boxsize = lya_file['Header'].attrs['box']/1000
z= 1.92
res_in_cmp = (100 /cosmo.H(z).value)*(1+z) * cosmo.h
num_pix = res_in_cmp / (boxsize/all_flux.shape[1])
coarse_flux = scipy.ndimage.gaussian_filter1d(all_flux , sigma=num_pix, axis=1, mode='wrap')
all_flux= coarse_flux
#all_flux= all_flux[:378]

ps_central = _powerspectrum(all_flux, axis=-1)
CNR = 2*np.exp(np.random.normal(0.5323449534337695, 0.36223889354351285, size=all_flux.shape[0]))
CE = 0.24*CNR**(0.86)

noisy_flux,delta= add_cont_error(CE, all_flux, spec_num=-1, u_delta=0.6, l_delta=-0.6)

noisy_flux, dn = add_noise(CNR, noisy_flux)

ps_n= _powerspectrum(noisy_flux, axis=-1)

diff = rmse(ps_central,ps_n)
dn_ds= (np.mean(dn)/diff)

F_ps= [power_spec[0]**2*dn_ds**2,power_spec[1]**2*dn_ds**2,power_spec[2]**2*dn_ds**2,power_spec[3]**2*dn_ds**2]
