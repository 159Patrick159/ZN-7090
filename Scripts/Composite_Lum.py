############################### HEADER ###############################
# Author: Patrick Sandoval                                           #
# Date: 07-03-2023                                                   #
######################################################################
# In the following script we will look at the composite luminosity   #
# and temperature of Morag, Waxman and Sapir model for type II SN    #
# we will look how each component of the luminosity contribute to    #
# the observed luminosity                                            #
######################################################################
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from Helpers import Nice_Plots

Nice_Plots()

def Model(t, R, vs, M, fp, w):
    '''Uses the extended model from sapir & waxman to get the 
    flux density in the blue band'''
    from Helpers import FluxWave
    import astropy.units as u
    # Compute the early T and L from S&W
    kappa = 1
    Trw = 1.61 * (vs**2*t**2/(fp*M*kappa))**0.027 *R**(1/4) / kappa**(1/4) *t**(-1/2) # eV
    Lrw = 2.0 * 10**42 *(vs*t**2/(fp*M*kappa))**(-0.086) * (vs**2*R)/kappa  # erg/s
    Trw = Trw*u.eV
    Lrw = Lrw*u.erg/u.s
    Msun = 1.989 * 10**33 # g
    #M = M*Msun
    E = (vs*10**(8.5)/1.05/fp**(-2*0.191))**2*M
    # Calculate envelope mass
    Menv = fp**2*M/(fp**2+1)
    #####################################################################################
    # Apply supression factor to Lbol at later times
    # Compute transparency time tr
    # Comments: To compute the transparency time  we need to know the envelope mass,
    # this is a very hard paremter to get because it depends on the progenitors density
    # structure. However, from SW they provide bounds for such quatity 0.1< Mc/Menv< 10
    # and for convective effective envelopes we have that fp = sqrt(Mc/Menv), hence 
    # tr = 13 * fp**(0.191/2) * (Menv)**(3/4) * (M/Menv)**(1/4) * (E/10**(51))**(-1/4) # days
    tr = 19.5*(kappa*Menv/vs)**(1/2) #days
    Lsup = Lrw*0.94*np.exp(-(1.67*t/tr)**0.8) # erg/s

    # Compute planar luminosity
    # Transform time from days to h
    th = t*24 # hr
    Lp = 2.974*10**(42)*(R**(0.462)*vs**(0.602))/(fp*M*kappa)**(0.0643)*(R**2)/(kappa)*th**(-4/3) # erg/s
    Lp = Lp * u.erg/u.s
    # Define composite Luminosity
    Lc = Lp + Lsup
    
    # Compute planar temperature
    Tp = 6.937*(R**(0.1155)*vs**(0.1506))/(fp**(0.01609)*M**(0.01609)*kappa**(0.2661))*th**(-1/3) # eV
    Tp = Tp*u.eV
    # Get composite temperature by taking the lowest temperature values
    Tc = []
    for i in range(len(Tp)):
        if Tp[i].value < Trw[i].value:
            Tc.append(Tp[i].value)
        else:
            Tc.append(Trw[i].value)
    # Apply color correction to composite temperature
    Tc = np.array(Tc)
    # Testing temperature behaviour
    # Tc = Trw 
    Tc = Tc*u.eV
    # Convert temperature from eV to K
    kB = 8.617333262*1e-5*u.eV/u.K #eV/K
    Tc = Tc/kB
    Trw = Trw/kB

    # Define redshift
    z = 0.08739904944703399

    # Define distance
    # Dl = 377.044*1e6 * u.pc # In case of using apparent magnitude
    # Dl = 10 * u.pc # If using absolute magnitudes
    ############# VALUES FOR SN2018fif #############
    # Dl = 72*1e6 *u.pc
    Dl = 70.39600*1e6*u.pc
    z = 0

    Dl = Dl.to(u.cm) # Convert to cm
    # L: erg/s, T: K, w: cm, Dl: cm
    fpred = FluxWave(Lc,Tc,w,Dl,z) # My function returns erg/s/cm2/cm
    # fpred_sub = FluxWave(Lsup,Trw,w,Dl,z)
    # Convert to erg/s/cm2/A
    fpred = fpred.to(u.erg/u.s/u.cm**2/u.AA)
    # fpred_sub = fpred_sub.to(u.erg/u.s/u.cm**2/u.AA)
    # return(fpred_sub.value)
    # return(fpred.value)
    return(fpred.value/2,Lc.value,Tc.value)


# Define wavelengths for each band
B_wave = 0.445*u.um #um
V_wave = 0.551*u.um #um
i_wave = 0.806*u.um #um

################### ZN7090 Parameters ################### 
# R13 = 7.34031437
# vs8 = 0.48872504
# M = 10.22205346
# fp = 4.99544473
alpha = 0.1
#########################################################

################### SN2018fif Parameters ################### 
Rsun_model = 804 # Solar radii
vs8 = 0.817 # cm/s
M = 6.7 # Solar mass
fp = 1.86 # Dimensionless

Rsun = 6.957e+10 #cm
R_model = Rsun*Rsun_model # cm
R13 = R_model/1e13
############################################################
# Create time array
t = np.linspace(0.2,15,100)

# Get composite luminosities
Fc, Lc, Tc = Model(t,R13,vs8,M,fp,B_wave.to(u.cm))

fig, ((a0,a1),(a2,a3)) = plt.subplots(figsize=(8,6),ncols=2,nrows=2)

a0.plot(t,Fc,label='Composite Model',c='cyan')
# a0.plot(t,Fsw,label='Sapir & Waxman Supression',ls='--',c='k')
# a0.plot(t,Fp,label='Sapir, Morag, Waxman Extension',ls='--',alpha=0.5,c='r')
a0.set_ylabel(r"F$_\lambda$ (erg/s/cm$^2$/$\AA$)")
a0.grid()
h, l = a0.get_legend_handles_labels()

a1.plot(t,Lc,label='Composite Luminosity',c='cyan')
# a1.plot(t,Lsup,label='Supressed Luminosity',ls='--',c='k')
# a1.plot(t,Lp,label='Late Planar Lumninosity',ls='--',alpha=0.8,c='r')
a1.set_ylabel("Luminosity (erg/s)")
a1.grid()


a2.plot(t,Tc,label='Composite Luminosity',c='cyan')
# a2.plot(t,Trw,label='Early Temperature',ls='--',c='k')
# a2.plot(t,Tp,label='Planar Temperature',ls='--',alpha=0.8,c='r')
a2.set_ylabel("Temperature (K)")
a2.set_xlim([0,4])
a2.grid()
a3.axis('off')
a3.legend(h,l,loc='center')

plt.show()


