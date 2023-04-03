import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
