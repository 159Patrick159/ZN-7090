import emcee
import numpy as np
import astropy.units as u
from Helpers import FluxWave
# Call current wavelength from ZN7090MCMC script
from ZN7090_MCMC_Log import w

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
        R13, vs8, M, fp = theta

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
        ymodel = MCMCSimult3(theta,x)
        Lnlike = -0.5*np.sum((y-ymodel)**2 / yerr**2)
        return Lnlike


def ZN7090_lnprior3(theta):
        R, vs, M, fp  = theta
        # Convert quantities into comparable values
        # Rsun = (R*1e13)/(6.957e+10) # cm -> Rsun
        # Prior are the same as the ones as Ido's paper
        Rlow = 200*6.957e+10/1e+13
        Rhigh = 3000*6.957e+10/1e+13
        if np.logical_and(R>=Rlow,R<=Rhigh):
        # if np.logical_and(R>=2,R<=20):
            if np.logical_and(vs>=0.3,vs<=3):
                if np.logical_and(M>=1.2, M<=25):
                    if np.logical_and(fp > 0,fp<= np.sqrt(10)):
                        return 0.0
        return(-np.inf)

def lnprob3(theta,x,y,yerr):
        lp = ZN7090_lnprior3(theta)
        if lp == 0.0:
            return lp + lnlike3(theta,x,y,yerr)
        return -np.inf

def SimultMainMCMC(p0,nwalkers,niter,ndim,lnprob,data,pool,nthreads):
        from time import time
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob3, args=data, pool=pool, threads=nthreads)
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
        mod = One_Model(domain,*i,w)
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model, spread

def Sample_Walker_Model(sampler,x,w,N):
    from Helpers import One_Model
    m = []
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=N)]:
        m.append(One_Model(x,*theta,w))
    return(m)