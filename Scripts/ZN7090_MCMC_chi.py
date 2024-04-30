################################# HEADER #################################
# This python script implements a MCMC fitting routine on the early      #
# light curves of ZN7090. This script will read in the corrected LC's    #
# from Mathews work, it will convert them to monochromatic luminosities  #
# and will use the model from Rabinak, Waxmann to estimate the explosion #
# and progenitor parameters.                                             #
##########################################################################
#%%
import pandas as pd
import numpy as np
import emcee
import matplotlib.pyplot as plt
import astropy.units as u
from Helpers import Nice_Plots, day_mjd, m2M, FluxDensity, BBflux

B_wave = 0.445*u.um #um
V_wave = 0.551*u.um #um
i_wave = 0.806*u.um #um

# Convert wavelegths from um to cm
B_wave = B_wave.to(u.cm)
V_wave = V_wave.to(u.cm)
i_wave = i_wave.to(u.cm)
Bands = {"B":B_wave,"V":V_wave,"I":i_wave}

# Define wavelenght for single model
band = "B"
w = Bands[band]

############################################################################

# Call Nice_Plots
Nice_Plots()

# Read Mathew's photometry data
ZN7090_early = pd.read_csv("KSP-ZN7090_LC_20220309_correct.csv")
# Collect all dates
dates = np.array(ZN7090_early['date[MJD]'],dtype='float')

# Get band index
bands = ZN7090_early["band"]
loc_B = np.where(bands=="B")
loc_V = np.where(bands=="V")
loc_I = np.where(bands=='I')

#####################################################################################
############################# PREPARE DATA FOR ANALYSIS #############################
#####################################################################################

# Index the corrected magnitude arrays on the bands of interest
all_mags = np.array(ZN7090_early["M_E"])
all_mag_err = np.array(ZN7090_early["M_E_err"])
B_early = all_mags[loc_B]
B_early_err = all_mag_err[loc_B]
V_early = all_mags[loc_V]
V_early_err = all_mag_err[loc_V]
I_early = all_mags[loc_I]
I_early_err = all_mag_err[loc_I]

# Get the respective dates
B_dates = dates[loc_B]
V_dates = dates[loc_V]
I_dates = dates[loc_I]

B_dates = B_dates[3:]
V_dates = V_dates[3:]
I_dates = I_dates[3:]

# Elimintate non detections
Bapp = B_early[3:]
BappErr = B_early_err[3:]
Vapp = V_early[3:]
VappErr = V_early_err[3:]
Iapp = I_early[3:]
IappErr = I_early_err[3:]

print("Data loaded succesfully!")
print()
print("Converting apparent magnitudes to absolute...")

#############################################################################

# Convert apparent magntiudes to absolute
print("Performing K-corrections...")
z = 0.08739904944703399
Babs, BabsErr = m2M(Bapp,BappErr,z)
Vabs, VabsErr = m2M(Vapp,VappErr,z)
Iabs, IabsErr = m2M(Iapp,IappErr,z)

print("Conversions completed!")
print()
#############################################################################

# Convert to monochromatic fluxes

# Define zeropoints flux densities
fwaveB = 632*1e-11 #erg/s/cm2/A
fwaveV = 363.1*1e-11 # erg/s/cm2/A
fnuB = 4063*1e-23 # Jy -> erg/s/cm2/Hz
fnuV = 3636*1e-23 # Jy -> erg/s/cm2/Hz
fnui = 3631*1e-23 # Jy -> erg/s/cm2/Hz

print("Converting to monochromatic flux densities")
print("B-band...")
fBpc, fBErrpc,fBErrpc2 = FluxDensity(Babs,BappErr,B_wave,fwave0=fwaveB) # erg/s/cm2/A
fB, fBErr,fBErr2 = FluxDensity(Bapp,BabsErr,B_wave,fnu0=fnuB)

print("V-band...")
fVpc, fVErrpc, fVErrpc2 = FluxDensity(Vabs,VabsErr,V_wave,fwave0=fwaveV)
fV, fVErr, fVErr2 = FluxDensity(Vapp,VappErr,V_wave,fwave0=fwaveV) # erg/s/cm2/cm

print("I-band...")
fIpc, fIErrpc, fIErrpc2 = FluxDensity(Iabs,IabsErr,i_wave,fnu0=fnui) # erg/s/cm2/cm
fI, fIErr, fIErr2 = FluxDensity(Iapp,IappErr,i_wave,fnu0=fnui) #

print()
print("Conversion complete!")
print()

# Plot light curves to see their scale
fig, ax = plt.subplots(figsize=(8,6))
ax.set_title("Monochromatic Light Curves",fontsize=16)
ax.errorbar(B_dates,fB,yerr=fBErr,fmt='bo')
ax.errorbar(V_dates,fV,yerr=fVErr,fmt='go')
ax.errorbar(I_dates,fI,yerr=fIErr,fmt='ro')
ax.set_xlabel("Modified Julian Date")
ax.set_ylabel("Monochromatic Flux (erg/s/cm$^2$/$\AA$) Observer")

ax2 = ax.twinx()
ax2.errorbar(B_dates,fBpc,yerr=fBErrpc,fmt='bo')
ax2.errorbar(V_dates,fVpc,yerr=fVErrpc,fmt='go')
ax2.errorbar(I_dates,fIpc,yerr=fIErrpc,fmt='ro')
ax2.set_ylabel("Monochromatic Flux (erg/s/cm$^2$/$\AA$) 10pc")
plt.savefig("Plots/Monochromatic_LCs.png")

####################################################################################
# For the model we will be fixing our source at 10pc which mean our monochromatic  #
# flux densities are representing that of a source 10pc which means that our model #
# needs to be adjusted for a bb source at 10pc                                     #
####################################################################################

# Convert magnitudes from erg/s/cm2/A to erg/s/cm2/cm
# fB *= 1e-8
# fV *= 1e-8
# fI *= 1e-8
# fBErr *= 1e-8
# fVErr *= 1e-8
# fIErr *= 1e-8
# Define bands color format
BandC = {"B":"b","V":"g","I":"r"}
BandFMT = {"B":"bo","V":"go","I":"ro"}

# Use curve_fit to estimate p0 for MCMC
from scipy.optimize import curve_fit
from Helpers import FluxWave, Fnu2Fwave

# Define time of explosion
t0 = B_dates[0]-0.2320
t0Err = 0.12

# Have all time arrays with respect to explosion time
B_dates -= t0
V_dates -= t0
I_dates -= t0

# Set sampling interval
tsample = 12


#%%
if __name__ == '__main__':
    def Simult3LC(t,R13,vs8,M,fp):
        ''' 
        All input parameters must follow these equalities for model
        and have the correct units
        R =  R13 * 1e13 # cm
        vs = vs8 * 1e8.5 # cm/s
        M # Msun
        fp # Dimensionless
        '''
        # Import one LC model from Helpers
        from Helpers import One_Model

        # Split time array into the three lc data
        Blen = len(B_dates[np.where(B_dates<tsample)])
        Vlen = len(V_dates[np.where(V_dates<tsample)])
        Ilen = len(I_dates[np.where(I_dates<tsample)])
        tB = t[ : Blen]
        tV = t[Blen : Blen + Vlen]
        tI = t[Blen + Vlen : Blen + Vlen+ Ilen]

        # Predict the flux densities for each band
        fBpred = One_Model(tB,R13,vs8,M,fp,B_wave) # erg/s/cm2/A
        fVpred = One_Model(tV,R13,vs8,M,fp,V_wave) # erg/s/cm2/A
        fIpred = One_Model(tI,R13,vs8,M,fp,i_wave) # erg/s/cm2/A

        # Concatenate the arrays
        fpred = np.concatenate((fBpred,fVpred,fIpred))
        return fpred

    # Create concatenated data for sim fitting.
    full_data = np.concatenate((fB[np.where(B_dates<tsample)],fV[np.where(V_dates<tsample)],fI[np.where(I_dates<tsample)]))
    full_domain = np.concatenate((B_dates[np.where(B_dates<tsample)],V_dates[np.where(V_dates<tsample)],V_dates[np.where(V_dates<tsample)]))
    full_errs = np.concatenate((fBErr2[np.where(B_dates<tsample)],fVErr2[np.where(V_dates<tsample)],fIErr2[np.where(I_dates<tsample)]))

    from Helpers import chi_square_reduced
    # Fit data with least-square
    popt3, pcov3 = curve_fit(Simult3LC,full_domain,full_data,sigma=full_errs,p0=[3,1,3,1],maxfev=2000,bounds=((1.2,0.3,2.5,np.sqrt(1/3)),(10.5,3,25,np.sqrt(10))))

    # popt3, pcov3 = curve_fit(Simult3LC,full_domain,full_data,sigma=full_errs,p0=[10,2,5,3],maxfev=2000,bounds=((0,0,2,0),(np.inf,np.inf,np.inf,np.inf)))
    print("Simultaneous Least-Square fit parameters:",popt3)
    print(f"Reduced chi-square:"\
        ,chi_square_reduced(full_data,Simult3LC(full_domain,*popt3),full_errs,4.0))
    
    # Store lc data
    Bxdata = B_dates[np.where(B_dates<tsample)]
    Vxdata = V_dates[np.where(V_dates<tsample)]
    Ixdata = I_dates[np.where(I_dates<tsample)]

    Bydata = fB[np.where(B_dates<tsample)]
    Vydata = fV[np.where(V_dates<tsample)]
    Iydata = fI[np.where(I_dates<tsample)]

    Bdataerr = fBErr2[np.where(B_dates<tsample)]
    Vdataerr = fVErr2[np.where(V_dates<tsample)]
    Idataerr = fIErr2[np.where(I_dates<tsample)]

    # Generate domains for predicted values
    Bnice = np.linspace(np.min(Bxdata),np.max(Bxdata),500)
    Vnice = np.linspace(np.min(Bxdata),np.max(Bxdata),500)
    Inice = np.linspace(np.min(Bxdata),np.max(Bxdata),500)

    from Helpers import One_Model
    Bpred = One_Model(Bnice,*popt3,B_wave)
    Vpred = One_Model(Vnice,*popt3,V_wave)
    Ipred = One_Model(Inice,*popt3,i_wave)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.errorbar(Bxdata,Bydata,yerr=Bdataerr,fmt='bo')
    ax.errorbar(Vxdata,Vydata,yerr=Vdataerr,fmt='go')
    ax.errorbar(Ixdata,Iydata,yerr=Idataerr,fmt='ro')
    ax.plot(Bnice,Bpred,c='b')
    ax.plot(Vnice,Vpred,c='g')
    ax.plot(Inice,Ipred,c='r')
    ax.set_xlabel("Time since explosion (days)")
    ax.set_ylabel(r"Flux Density (erg/s/cm/$\AA$)")
    ax.set_title("ZN7090-2020f Least-Square Fit")
    plt.show()
    #%%
    #################################################################################################
    ################################### Simultaneous MCMC Fitting ###################################
    #################################################################################################
    # Define MCMC routine here
    from MCMC_Routines_chi import SimultMainMCMC,lnprob3,MCMCSimult3
    
    # Import needed libraries
    import corner
    from multiprocessing import Pool

    # Setup number of walkers and number or iterations
    nwalkers = 250
    niter = 500
    # Physically motivated parameters from curve_fit
    initial = popt3

    # # Try smaller radius 
    initial[0] = 9.0
    # Append f value to parameter
    print("Initial Guess:",initial)
    ndim = len(initial)
    # Define step size for walkers
    p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]

    # Define number of thread for multiprocessing
    nthreads = 8
    data = (full_domain,full_data,full_errs)

    # Use parallel computing for MCMC
    with Pool(processes=6) as pool:
        sampler, pos, prob, state = SimultMainMCMC(p0,nwalkers,niter,ndim,lnprob3,data,pool,nthreads)
        pool.close()

    # Get highest likelyhood parameters
    samples = sampler.flatchain
    theta_max  = np.array(samples[np.argmax(sampler.flatlnprobability)])
    chi_red = chi_square_reduced(full_data,MCMCSimult3(theta_max,full_domain),full_errs,4.0)
    print("Best Parameters:",theta_max)
    print(f"MCMC reduced chi-square"\
        ,chi_red)
    print(f"MCMC reduced chi-square"\
        ,chi_red)
    #%%
    # Look at chain behaviour
    chain = sampler.get_chain()
    labels = [r"$R$",r"$v_{s,8.5}$",r"$M$",r"$f_{\rho}$"]
    fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(chain[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(chain))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],range=[(15,19),(1.4,1.9),(1,25),(0.15,0.8)])
    #plt.savefig("Plots/MCMC_Results/CHIcorner_Relaxed.pdf")
    from IPython.display import display, Math
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))
    #%%
    # Hard code the parameters from corner plot
    # Original chi
    #theta_max = [10.401,2.115,10.760,0.344]
    # Modified chi
    theta_max2 = [17.796, 1.593, 11.136, 0.361]

    # theta_max =  [20.94740177 , 1.52524034 , 26.02497211 , 0.19238536,-1.68629237]
    chi_red = chi_square_reduced(full_data,MCMCSimult3(theta_max2,full_domain),full_errs,len(theta_max))
    print("Best Parameters:",theta_max2)
    print(f"MCMC reduced chi-square",chi_red)
    #%%
    from MCMC_Routines_chi import sample_walkers_w
    from matplotlib.ticker import MultipleLocator
    from MCMC_Routines_chi import Sample_Walker_Model
    # Compute resiuals 
    Bpred_data = One_Model(Bxdata,*theta_max2,B_wave)
    Vpred_data = One_Model(Vxdata,*theta_max2,V_wave)
    Ipred_data = One_Model(Ixdata,*theta_max2,i_wave)
    Bres = (Bpred_data-Bydata)
    Vres = (Vpred_data-Vydata)
    Ires = (Ipred_data-Iydata)

    # Sample Walker
    N = 200
    Bwalk = Sample_Walker_Model(sampler,Bnice,B_wave,N)
    Vwalk = Sample_Walker_Model(sampler,Vnice,V_wave,N)
    Iwalk = Sample_Walker_Model(sampler,Inice,i_wave,N)

    Bwalk_data = Sample_Walker_Model(sampler,Bxdata,B_wave,N)
    Vwalk_data = Sample_Walker_Model(sampler,Vxdata,V_wave,N)
    Iwalk_data = Sample_Walker_Model(sampler,Ixdata,i_wave,N)

    BpredMCMC = One_Model(Bnice,*theta_max2,B_wave)
    VpredMCMC = One_Model(Vnice,*theta_max2,V_wave)
    IpredMCMC = One_Model(Inice,*theta_max2,i_wave)

    ############### Uncomment if interested in posterior dist. ###############
    MedB, SprB = sample_walkers_w(Bnice,100,samples,B_wave)
    MedV, SprV = sample_walkers_w(Vnice,100,samples,V_wave)
    MedI, SprI = sample_walkers_w(Inice,100,samples,i_wave)
    SprB *= 1
    SprV *= 1
    SprI *= 1
    fig, (ax,ax1) = plt.subplots(figsize=(8,6),sharex=True,\
                            gridspec_kw={'height_ratios': [3, 1]},nrows=2)
    ax.fill_between(Bnice,MedB-SprB,MedB+SprB,color='b',alpha=0.2,label=r'$1\sigma$ Posterior Spread')
    ax.fill_between(Vnice,MedV-SprV,MedV+SprV,color='g',alpha=0.2)
    ax.fill_between(Inice,MedI-SprI,MedI+SprI,color='r',alpha=0.2)
    ax.fill_between(Bnice,MedB-SprB*2,MedB+SprB*2,color='b',alpha=0.2,label=r'$2\sigma$ Posterior Spread')
    ax.fill_between(Vnice,MedV-SprV*2,MedV+SprV*2,color='g',alpha=0.2)
    ax.fill_between(Inice,MedI-SprI*2,MedI+SprI*2,color='r',alpha=0.2)
    ax.fill_between(Bnice,MedB-SprB*3,MedB+SprB*3,color='b',alpha=0.2,label=r'$3\sigma$ Posterior Spread')
    ax.fill_between(Vnice,MedV-SprV*3,MedV+SprV*3,color='g',alpha=0.2)
    ax.fill_between(Inice,MedI-SprI*3,MedI+SprI*3,color='r',alpha=0.2)
    #ax.legend()

    ###########################################################################
    ax.errorbar(Bxdata,Bydata,yerr=Bdataerr,fmt='bo')
    ax.errorbar(Vxdata,Vydata,yerr=Vdataerr,fmt='go')
    ax.errorbar(Ixdata,Iydata,yerr=Idataerr,fmt='ro')

    ################### Uncomment if interesting in walker plot ###################
    # for i in range(len(Bwalk)):
    #     ax.plot(Bnice,Bwalk[i],c='b',alpha=0.06)
    #     ax.plot(Vnice,Vwalk[i],c='g',alpha=0.06)
    #     ax.plot(Inice,Iwalk[i],c='r',alpha=0.06)
    #     ax1.scatter(Bxdata,Bwalk_data[i]-Bydata,c='b',alpha=0.2,s=12)
    #     ax1.scatter(Vxdata,Vwalk_data[i]-Vydata,c='g',alpha=0.2,s=12)
    #     ax1.scatter(Ixdata,Iwalk_data[i]-Iydata,c='r',alpha=0.2,s=12)
    ###############################################################################
    ax.plot(Bnice,BpredMCMC,c='b',ls='-.')
    ax.plot(Vnice,VpredMCMC,c='g',ls='-.')
    ax.plot(Inice,IpredMCMC,c='r',ls='-.')
    # ax.plot(Bnice,MedB,c='b',ls='-.')
    # ax.plot(Vnice,MedV,c='g',ls='-.')
    # ax.plot(Inice,MedI,c='r',ls='-.')
    ax.yaxis.set_minor_locator(MultipleLocator(.05e-16))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax1.set_xlabel("Time since explosion (days)",fontsize=16)
    ax.set_ylabel(r"Flux Density (erg/s/cm/$\AA$)",fontsize=16)
    ax.set_title("ZN7090-2020f MCMC Fit",fontsize=18)
    ax.text(0.5,1.85e-16,rf"$\chi^2$/d.o.f= {round(chi_red,2)}",fontsize=12)
    ax.grid(ls='--')

    ax1.scatter(Bxdata,Bres,c='b')
    ax1.scatter(Vxdata,Vres,c='g')
    ax1.scatter(Ixdata,Ires,c='r')
    # ax1.set_ylabel("Residuals",fontsize=16)    
    # ax1.axhline(y=0,ls='--',c='k')
    ax1.yaxis.set_minor_locator(MultipleLocator(.5e-17))
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.grid(ls='--')
    plt.tight_layout()
    #plt.savefig("Plots/MCMC_Results/CHILC_Relaxed.pdf")
    plt.show()