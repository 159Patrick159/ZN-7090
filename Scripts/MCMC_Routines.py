############################### HEADER ###############################
# This python script contains the MCMC routines used to extract      #
# the progenitor parameters for ZN7090 through a Sapir & Waxman      #
# early light curve emission model. This script contains a single    #
# lightcurve fittting routine as well as a simultanous triple light  #
# curve fitting routine.                                             #
######################################################################

import emcee
import numpy as np
import astropy.units as u
from Helpers import FluxWave
# Call current wavelength from ZN7090MCMC script
from ZN7090_MCMC_Log import w

################################################################################################
############################### Single Light Curve Fitting Model ###############################
################################################################################################

# Define wavelenght for single model, must change this if fitting different lc

def SingleLCModel(theta,t):
    import astropy.units as u
    # Define model from Sapir & Waxman 2017
    # Extract parameters
    R, vs, M, fp = theta 
    kappa = 1 # From theory paper opacity is constant for early emission k = k34 * 0.34
    # Compute luminosity and temperature
    Lrw = 2.0 * 10**42 * np.abs(vs*t**2/(fp*M*kappa))**(-0.086) * (vs**2*R)/kappa  # erg/s
    Trw = 1.61 *np.abs(vs**2*t**2/(fp*M*kappa))**0.027 *np.abs(R)**(1/4) / (np.abs(kappa)**(1/4)) *t**(-1/2) # eV
    Trw = Trw*u.eV
    Lrw = Lrw*u.erg/u.s

    # Convert temperature from eV to K
    kB = 8.617333262e-5*u.eV/u.K #eV/K
    Trw = Trw/kB
    
    # Define redshift
    z = 0.08739904944703399

    # Compute the predicted monochromatic flux density at 10pc
    # Dl = 10 #pc IMPORTANT to change if using aparent magnitudes!!!
    Dl = 377.044*1e6 *u.pc # In case of using apparent magnitude
    Dl = Dl.to(u.cm) # pc -> cm

    # L: erg/s, T: K, w: cm, Dl: cm
    fpred = FluxWave(Lrw,Trw,w,Dl,z) # My function returns erg/s/cm2/cm
    # Conver to erg/s/cm2/A
    fpred = fpred.to(u.erg/u.s/u.cm**2/u.AA)

    return(fpred.value)

def Composite_Single_Model(theta, t):
    '''Uses the extended model from sapir & waxman to get the 
    flux density in the blue band'''
    from Helpers import FluxWave
    # Compute the early T and L from S&W
    # Extract parameters
    R, vs, M, fp = theta
    kappa = 1
    Trw = 1.61 * (vs**2*t**2/(fp*M*kappa))**0.027 *R**(1/4) / kappa**(1/4) *t**(-1/2) # eV
    Lrw = 2.0 * 10**42 *(vs*t**2/(fp*M*kappa))**(-0.086) * (vs**2*R)/kappa  # erg/s
    Trw = Trw*u.eV
    Lrw = Lrw*u.erg/u.s

    # Apply supression factor to Lbol at later times
    # Compute transparency time tr
    tr = 19.5*(kappa*M/vs)**(0.5) # days
    Lsup = Lrw*0.94*np.exp(-(1.67*t/tr)**0.8)

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
    # Define redshift
    z = 0.08739904944703399
    # Tc = Trw 
    Tc = Tc*u.eV
    # Convert temperature from eV to K
    kB = 8.617333262*1e-5*u.eV/u.K #eV/K
    Tc = Tc/kB


    # Define distance
    Dl = 377.044*1e6 * u.pc # In case of using apparent magnitude
    # Dl = 10 * u.pc # If using absolute magnitudes
    # Dl = 72*1e6 *u.pc
    Dl = Dl.to(u.cm) # Convert to cm
    # L: erg/s, T: K, w: cm, Dl: cm
    fpred = FluxWave(Lc,Tc,w,Dl,z) # My function returns erg/s/cm2/cm
        
    # Convert to erg/s/cm2/A
    fpred = fpred.to(u.erg/u.s/u.cm**2/u.AA)
    return(fpred.value)

def lnlike(theta,x,y,yerr):
    #ymodel = SingleLCModel(theta,x)
    ymodel = Composite_Single_Model(theta,x)
    Lnlike = -0.5*np.sum((y-ymodel)**2 / yerr**2)
    return Lnlike

def ZN7090_lnprior(theta):
    R, vs, M, fp = theta
    # Convert quantities into comparable values
    #Rsun = (R*1e13)/(6.957e+10) # cm -> Rsun
    # Prior are the same as the ones as Ido's paper
    if np.logical_and(R>=1.3914,R<=10.435):
        if np.logical_and(vs>=0.5,vs<=3.0):
            if np.logical_and(M>=2, M<=25):
                if np.logical_and(fp>= np.sqrt(1/3),fp <= np.sqrt(10)):
                    return 0.0
    return(-np.inf)

def lnprob(theta,x,y,yerr):
    lp = ZN7090_lnprior(theta)
    if lp == 0.0:
        return lp + lnlike(theta,x,y,yerr)
    return -np.inf

def SingleMainMCMC(p0,nwalkers,niter,ndim,lnprob,data,pool,nthreads):
    from time import time
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool, threads=nthreads)
    start = time()
    print("Print Running Burn-in...")
    p0, _, _ = sampler.run_mcmc(p0,100)

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter,progress=True)
    end = time()

    print(f"MCMC took {end-start:.3f} seconds")
    print("Mean Acceptance:",np.mean(sampler.acceptance_fraction))
    return sampler, pos, prob, state


def sample_walkers(domain,nsamples,flattened_chain):
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    smooth_x = np.linspace(np.min(domain),np.max(domain),500)
    for i in thetas:
        #mod = SingleLCModel(i,smooth_x)
        mod = Composite_Single_Model(i,smooth_x)
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model, spread, smooth_x

################################################################################################
############################### Triple Light Curve Fitting Model ###############################
################################################################################################

def MCMCSimult3(theta,t):
        ''' 
        All input parameters must follow these equalities for model
        and have the correct units
        R =  R13 * 1e13 # cm
        vs = vs8 * 1e8.5 # cm/s
        M # Msun
        fp # Dimensionless
        '''
        # Import 1LC model from Helpers
        from Helpers import One_Model
        from ZN7090_MCMC_Log import B_dates, V_dates, I_dates, tsample, B_wave, V_wave, i_wave
        # Extract parameters
        R13, vs8, M, fp, log_f = theta

        # Define wavelengths for each band

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
    
def lnlike3(theta,x,y,yerr):
        '''
        Likelihood function here is a simple chi-square
        '''
        ymodel = MCMCSimult3(theta,x)
        Lnlike = -0.5*np.sum((y-ymodel)**2 / yerr**2)
        return Lnlike

def lnlikeG(theta,x,y,yerr):
        '''
        This likelihood function accounts for errors in
        the data and errors in the model itself.
        '''
        log_f = theta[-1]
        model = MCMCSimult3(theta,x)
        # This term accounts for the total variance due to errors in data
        # and errors in the model
        sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
        return (-0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2)))


def ZN7090_lnprior3(theta):
        R, vs, M, fp, log_f = theta
        # Convert quantities into comparable values
        # Rsun = (R*1e13)/(6.957e+10) # cm -> Rsun
        # Prior are the same as the ones as Ido's paper
        Rlow = 200*6.957e+10/1e+13
        Rhigh = 1500*6.957e+10/1e+13
        if np.logical_and(R>=Rlow,R<=Rhigh):
            if np.logical_and(vs>=0.5,vs<=3):
                if np.logical_and(M>=2, M<=25):
                    if np.logical_and(fp >= np.sqrt(1/3), fp<= np.sqrt(10)):
                        if np.logical_and(log_f > -3, log_f < 3):
                            return 0.0
        return(-np.inf)

def lnprob3(theta,x,y,yerr):
        lp = ZN7090_lnprior3(theta)
        if lp == 0.0:
            return lp + lnlikeG(theta,x,y,yerr)
        return -np.inf

def SimultMainMCMC(p0,nwalkers,niter,ndim,lnprob,data,pool,nthreads):
        '''
        Performs a simultaneous MLE fitting on multi-band emissions of supernova ZN7090
        '''
        from time import time
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool, threads=nthreads)
        start = time()
        print("Print Running Burn-in...")
        p0, _, _ = sampler.run_mcmc(p0,100,progress=True)
        sampler.reset()
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter,progress=True)
        end = time()

        print(f"MCMC took {end-start:.3f} seconds")
        print("Mean Acceptance:",np.mean(sampler.acceptance_fraction))
        return sampler, pos, prob, state

def sample_walkers_w(domain,nsamples,flattened_chain,w):
    from Helpers import One_Model
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        #mod = SingleLCModel(i,smooth_x)
        i2 = i[:-1]
        mod = One_Model(domain,*i2,w)
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model, spread

def Sample_Walker_Model(sampler,x,w,N):
    from Helpers import One_Model
    m = []
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=N)]:
        theta2 = theta[:-1]
        m.append(One_Model(x,*theta2,w))
    return(m)