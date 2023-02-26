import numpy as np
import matplotlib.pyplot as plt


########################### Luminosity Distance ###########################
# Define cosmological density parameters and H0 as mathew did
import astropy.units as u
from astropy.constants import c
print("Computing luminosity distance to SN 2018fif by integrating cosmological parameters...")
z = 0.017189 # SN 2018fif

OM = 0.27
OK = 0.00
OL = 0.73
H0 = (74.2*u.km/u.s/u.Mpc).to(1/u.s)

import scipy.integrate as integrate
# Numerically integrate for the comoving distance at z
def func(z,M,K,L):
    return(1/np.sqrt(M*(1+z)**3 + K*(1+z)**2 + L))
DC = c/H0 * integrate.quad(func,0,z,args=(OM,OK,OL))[0]
# Compute comovin distance
DL = DC*(1+z)
print("Luminosity Distance for SN 2018fif",DL.to(u.Mpc),'Mpc')
print()


# Compute luminosity distance to SN 2018fif through distance modulus
# coefficient provided by paper
print("Now computing luminosity distance using distance modulus from paper...")
mu = 34.31
proper_dl = 10 * 10**(mu/5) * u.pc
print("Luminosity Distance for SN 2018fif:",proper_dl.to(u.Mpc))

########################### Compute flux densities ###########################

t = np.linspace(0.2,30,100)
# Define parameters from their corner plot from sn2018fif
kappa34 = 1 # g/cm2
Rsun_model = 804 # Solar radii
vs85 = 0.817 # cm/s
Msun = 6.7 # Solar mass
fp = 1.86 # Dimensionless

# Convert radius for model to work with
Rsun = 6.957e+10 #cm
R_model = Rsun*Rsun_model # cm
R13 = R_model/1e13
B_wave = 0.4361*u.um # um
B_wave = B_wave.to(u.cm) # convert to cm

def CF_Model(t,kappa, R, vs, M, fp):
    from Helpers import FluxWave
    # Define model from Sapir & Waxman 2017
    # Slice time array up until ts (fitting interval)
    # Compute luminosity and temperature
    Trw = 1.61 * np.sign(vs**2*fp*M*kappa)*np.abs(vs**2*t**2/(fp*M*kappa))**0.027 * np.sign(R)*np.abs(R)**(1/4) / (np.sign(kappa)*np.abs(kappa)**(1/4)) *t**(-1/2) # eV
    Lrw = 2.0 * 10**42 * np.sign(vs*fp*M*kappa)*np.abs(vs*t**2/(fp*M*kappa))**(-0.086) * (vs**2*R)/kappa  # erg/s
    Trw = Trw*u.eV
    Lrw = Lrw*u.erg/u.s
    # Convert temperature from eV to K
    kB = 8.617333262*1e-5*u.eV/u.K #eV/K
    Trw = Trw/kB

    # Compute the predicted monochromatic flux density at 10pc
    # Dl = 10 #pc IMPORTANT to change if using aparent magnitudes!!!
    # Dl = 377.044*1e6 # In case of using apparent magnitude
    # Dl = 70.396*1e6 * u.pc # pc for sn2018fif my estimate through integration of z
    Dl = 72.777*1e6 *u.pc # pc for sn2018fif using distance modulus = 34.31
    Dl = Dl.to(u.cm) # Convert to cm
    # L: erg/s, T: K, w: cm, Dl: cm
    fpred = FluxWave(Lrw,Trw,B_wave,Dl,z) # My function returns erg/s/cm2/cm

    ####################### Uncomment if using SNAP functions #######################
    # fpred = BBflux(Lrw,Trw,w,z,Dl) # Return erg/s/cm2/Hz                          #
    # fpred = Fnu2Fwave(fpred,w) # Returns in erg/s/cm2/cm                          #
    #################################################################################

    # Convert to erg/s/cm2/A
    return(fpred.to(u.erg/u.s/u.cm**2/u.AA))

data = CF_Model(t,kappa34,R13,vs85,Msun,fp)
plt.plot(t,data)
plt.grid()
plt.show()







