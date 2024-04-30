def Martinez_cooling_Asymmetric1(Bmags,Vmags,BmagsErr,VmagsErr):
    
    '''Returns the bolometric magntiudes, their respective uncertainties and the BC given Martinez et al.
    bolometric corrections for a SNe in its cooling phase. Its input must be two arrays, B and V magnitudes
    respectively and their uncertainties. The uncertainties are found through a Gaussian sampling method where 
    we assume the distribution is symmetric, thus we use the std as a measure of spread.'''
    
    import numpy as np
    
    # Initialize the coefficients for the BC with respective uncertainties
    sigma = 0.12
    c0, c1, c2, c3, c4 = -0.740, 4.472, -9.637, 9.075, -3.290
    
    # Define BC with coefficients
    BC  = lambda x: c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
    
    # Initialize empty arrays
    mbol_med = []
    mbol_Uerr = []
    mbol_Lerr = []
    
    color_mean = []
    color_std = []
    
    BC_med = []
    BC_Uerr = []
    BC_Lerr = []
    
    # We randomly sample 1000 points from  each magnitide point and compute te bolometric mag
    for i in range(len(Bmags)):
        sB_mag = np.random.normal(loc=Bmags[i],scale=BmagsErr[i],size=1000)
        sV_mag = np.random.normal(loc=Vmags[i],scale=VmagsErr[i],size=1000)
        
        s_color = sB_mag - sV_mag
        color_mean.append(np.mean(s_color))
        color_std.append(np.std(s_color))
        
        s_BC = BC(s_color)
        BC_med.append(np.median(s_BC))
        BC_uq = np.percentile(s_BC,75)
        BC_lq = np.percentile(s_BC,25)
        BC_Uerr.append(BC_uq-np.median(s_BC))
        BC_Lerr.append(np.median(s_BC) - BC_lq)
        
        s_mbol = BC(s_color) + sB_mag
        mbol_med.append(np.median(s_mbol))
        mbol_uq = np.percentile(s_mbol,75)
        mbol_lq = np.percentile(s_mbol,25)
        mbol_Uerr.append(mbol_uq - np.median(s_mbol))
        mbol_Lerr.append(np.median(s_mbol) - mbol_lq)
        
    return(mbol_med,mbol_Uerr,mbol_Lerr,color_mean,color_std,BC_med,BC_Uerr,BC_Lerr)

def Martinez_cooling_Asymmetric2(Bmags,Vmags,BmagsErr,VmagsErr,threshold_err,data_range):
    
    '''Returns the bolometric magntiudes, their respective uncertainties and the BC given Martinez et al.
    bolometric corrections for a SNe in its cooling phase. Its input must be two arrays, B and V magnitudes
    respectively and their uncertainties. The errorbars are calculated through a monte carlo simulation of
    Gaussian sampling, where the uncertainties are taken to be the IQR of the distribution of the bolometric
    magnitudes.'''
    
    import numpy as np
    
    # Initialize the coefficients for the BC with respective uncertainties
    sigma = 0.12
    c0, c1, c2, c3, c4 = -0.740, 4.472, -9.637, 9.075, -3.290
    
    # Define BC with coefficients
    BC  = lambda x: c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
    
    # Initialize empty arrays
    mbol_med = []
    mbol_UErr = []
    mbol_LErr = []
    
    color_mean = []
    color_std = []
    
    BC_med = []
    BC_UErr = []
    BC_LErr = []
    
    to_remove = []
    # We randomly sample 1000 points from  each magnitide point and compute te bolometric mag
    for i in range(len(Bmags)):
        sB_mag = np.random.normal(loc=Bmags[i],scale=BmagsErr[i],size=1000)
        sV_mag = np.random.normal(loc=Vmags[i],scale=VmagsErr[i],size=1000)
        
        s_color = sB_mag - sV_mag
        
        # Test if color is within range
        if np.mean(s_color) > data_range[1] or np.mean(s_color) < data_range[0]:
            # Test if color err is greater than threshold
            if np.std(s_color) > threshold_err:
                # Append index to remove in dates array
                to_remove.append(i)
                # Break for loop
                continue
                
        # Append if color satisfies the criterias
        color_mean.append(np.mean(s_color))
        color_std.append(np.std(s_color))
                
        
        s_BC = BC(s_color)
        BC_med.append(np.median(s_BC))
        BC_uq = np.percentile(s_BC,75)
        BC_lq = np.percentile(s_BC,25)
        BC_UErr.append(BC_uq - np.median(s_BC))
        BC_LErr.append(np.median(s_BC)-BC_lq)
        
        s_mbol = s_BC + sB_mag
        median = np.median(s_mbol)
        upper_quantile = np.percentile(s_mbol,75)
        lower_quantile = np.percentile(s_mbol,25)
        mbol_med.append(median)
        mbol_UErr.append(upper_quantile-median)
        mbol_LErr.append(median-lower_quantile)
        
    return(mbol_med,mbol_UErr,mbol_LErr,color_mean,color_std,BC_med,BC_UErr,BC_LErr,to_remove)

def Layman2Luminosity(mag1,mag2,mag1err,mag2err,c0,c1,c2,z,r,dist=None):
    
    '''Applies Laymans' bolometric correction of a second order polynomial
    to mag1 and mag2 to find the apparent bolometric correction. Consequently 
    it uses given redshift to compute the absolute magntiude and bolometric
    luminosity. Function uses IQR if dist is assymetric "A" or std if 
    distribution is symmetric "S". And r is the color range of the bolometric correction'''
    
    import numpy as np
    import scipy.integrate as integrate
    import astropy.units as u
    from astropy.constants import c
    
    # Define cosmological density parameters
    OM = 0.27
    OK = 0.00
    OL = 0.73
    
    # Define bolometric values for sun
    Lsun = 3.846e33
    Msun = 4.74
    
    # Define function for comoving distance computation
    def func(z,M,K,L):
        return(1/np.sqrt(M*(1+z)**3 + K*(1+z)**2 + L))
    
    # Perform numerical integration for Dc
    val = integrate.quad(func,0,z,args=(OM,OK,OL))

    # Using H = 67.5 (+/-) 0.5 km/s/Mpc
    H = (73.24*u.km/u.s/u.Mpc).to(1/u.s) # 1/s
    DC = c/H *val
    DL = DC*(1+z)
    Dl, Dl_err = (DL[0].to(u.pc)).value, DL[1]
    
    # Define 2nd order polynomial for BC
    BC = lambda x: c0 + c1*x + c2*x**2
    

    # Initialize empty arrays for magnitudes
    L_med = []
    L_UErr = []
    L_LErr = []
    L_mean = []
    L_Err = []

    # COMMENT THIS OUT IF YOU WISH TO IGNORE COLOR BOUNDS
    # This chunk of code will select the magnitudes
    # which are within the specified color range 
    # Where r[0] is the lower bound and r[1] is the upper bound
    mag1c = mag1[np.where(np.logical_and(mag1-mag2 >= r[0], mag1-mag2 <= r[1]))]
    mag2c = mag2[np.where(np.logical_and(mag1-mag2 >= r[0], mag1-mag2 <= r[1]))]
    mag1errc = mag1err[np.where(np.logical_and(mag1-mag2 >= r[0], mag1-mag2 <= r[1]))]
    mag2errc = mag2err[np.where(np.logical_and(mag1-mag2 >= r[0], mag1-mag2 <= r[1]))]

    mag1 = mag1c
    mag2 = mag2c
    mag1err = mag1errc
    mag2err = mag2errc
    ##################################################################################
    
    flag = False
    for i in range(len(mag1)):
        s_mag1 = np.random.normal(loc=mag1[i],scale=mag1err[i],size=1000)
        s_mag2 = np.random.normal(loc=mag2[i],scale=mag2err[i],size=1000)
        
        s_color = s_mag1-s_mag2
        s_BC = BC(s_color)
        s_mbol = s_BC + s_mag1
        s_Mbol = s_mbol - 5*np.log10(Dl/(10))
        s_Lbol = Lsun*100**((Msun - s_Mbol)/5) 
        
        
        # Noting that the distribution is assymetric we will use the median and 50% CI
        if dist == "S":
            L_mean.append(np.mean(s_Lbol))
            L_Err.append(np.std(s_Lbol))
            flag = True
            
        if dist == "A":
            L_med.append(np.median(s_Lbol))
            L_UErr.append(np.percentile(s_Lbol,75)-np.median(s_Lbol))
            L_LErr.append(np.median(s_Lbol) - np.percentile(s_Lbol,25))
        
    # Return Mean and Std if dist. is S
    if flag:
        return(L_mean,L_Err)
    
    # Return Med and IQR if dist. is A
    return(L_med,[L_LErr,L_UErr])


def Layman2Luminosity2(mag1,mag2,mag1err,mag2err,eff_range,c0,c1,c2,z,dist=None):
    
    '''Applies Laymans' bolometric correction of a second order polynomial
    to mag1 and mag2 to find the apparent bolometric correction. Consequently 
    it uses given redshift to compute the absolute magntiude and bolometric
    luminosity. Function uses IQR if dist is assymetric "A" or std if 
    distribution is symmetric "S".'''
    
    import numpy as np
    import scipy.integrate as integrate
    import astropy.units as u
    from astropy.constants import c
    
    # Define cosmological density parameters
    OM = 0.27
    OK = 0.00
    OL = 0.73
    
    # Define bolometric values for sun
    Lsun = 3.846e33
    Msun = 4.74
    
    # Define function for comoving distance computation
    def func(z,M,K,L):
        return(1/np.sqrt(M*(1+z)**3 + K*(1+z)**2 + L))
    
    # Perform numerical integration for Dc
    val = integrate.quad(func,0,z,args=(OM,OK,OL))

    # Using H = 67.5 (+/-) 0.5 km/s/Mpc
    H = (73.24*u.km/u.s/u.Mpc).to(1/u.s) # 1/s
    DC = c/H *val
    DL = DC*(1+z)
    Dl, Dl_err = (DL[0].to(u.pc)).value, DL[1]
    
    # Define 2nd order polynomial for BC
    BC = lambda x: c0 + c1*x + c2*x**2
    

    # Initialize empty arrays for magnitudes
    L_med = []
    L_UErr = []
    L_LErr = []
    L_mean = []
    L_Err = []
    bad_ones = []
    
    flag = False
    for i in range(len(mag1)):
        # Check if color is within effective range
        if mag1[i]-mag2[i] < eff_range[0] or mag1[i]-mag2[i] > eff_range[1]:
            bad_ones.append(i)
            continue
        
        s_mag1 = np.random.normal(loc=mag1[i],scale=mag1err[i],size=1000)
        s_mag2 = np.random.normal(loc=mag2[i],scale=mag2err[i],size=1000)
        
        s_color = s_mag1-s_mag2
        s_BC = BC(s_color)
        s_mbol = s_BC + s_mag1
        s_Mbol = s_mbol - 5*np.log10(Dl/(10))
        s_Lbol = Lsun*100**((Msun - s_Mbol)/5) 
        
        
        # Noting that the distribution is assymetric we will use the median and 50% CI
        if dist == "S":
            L_mean.append(np.mean(s_Lbol))
            L_Err.append(np.std(s_Lbol))
            flag = True
            
        if dist == "A":
            L_med.append(np.median(s_Lbol))
            L_UErr.append(np.percentile(s_Lbol,75)-np.median(s_Lbol))
            L_LErr.append(np.median(s_Lbol) - np.percentile(s_Lbol,25))
        
    # Return Mean and Std if dist. is S
    if flag:
        return(L_mean,L_Err,bad_ones)
    
    # Return Med and IQR if dist. is A
    return(L_med,[L_LErr,L_UErr],bad_ones)  
    
def Layman_BC(mag1,mag2,mag1err,mag2err,range_eff,c0,c1,c2,rms,dates):
    
    '''Applies Laymans' bolometric correction of a second order polynomial
    to mag1 and mag2 over the effective range provided. Use a Gaussian sampling
    method to find the distribution of the bolometric magnitudes and uses their
    IQR as  the uncertainty for the specifi mag'''
    
    import numpy as np
    # Define 2nd order polynomial for BC
    BC = lambda x: c0 + c1*x + c2*x**2
    
    # Compute only the colors that are within the specified range
    lower = range_eff[0]
    upper = range_eff[1]
    
    # Find the pair of magnitudes that are within the effective range
    test_colors = mag1 - mag2
    good_mags = []
    for i,color in enumerate(test_colors):
        if color <= upper and color >= lower:
            good_mags.append(i)
            
    # Only select the pair mags that are within the effective range of color
    # mag1 = mag1[good_mags]
    # mag2 = mag2[good_mags]
    
    # And the corresponding dates
    dates = dates[good_mags]
    
    color_med = []
    color_std = []
    
    BC_med = []
    BC_UErr = []
    BC_LErr = []
    
    mbol_med = []
    mbol_UErr = []
    mbol_LErr = []
    
    for i in range(len(mag1)):
        s_mag1 = np.random.normal(loc=mag1[i],scale=mag1err[i],size=1000)
        s_mag2 = np.random.normal(loc=mag2[i],scale=mag2err[i],size=1000)
        
        s_color = s_mag1-s_mag2
        color_med.append(np.median(s_color))
        color_std.append(np.std(s_color))
        
        s_BC = BC(s_color)
        BC_med.append(np.median(s_BC))
        BC_uq = np.percentile(s_BC,75)
        BC_lq = np.percentile(s_BC,25)
        BC_UErr.append(BC_uq-np.median(s_BC))
        BC_LErr.append(np.median(s_BC)-BC_lq)
        
        s_mbol = s_BC + s_mag1
        mbol_med.append(np.median(s_mbol))
        mbol_uq = np.percentile(s_mbol,75)
        mbol_lq = np.percentile(s_mbol,25)
        mbol_UErr.append(mbol_uq-np.median(s_mbol))
        mbol_LErr.append(np.median(s_mbol)-mbol_lq)
        
    return(mbol_med,mbol_UErr,mbol_LErr,color_med,color_std,BC_med,BC_UErr,BC_LErr,dates)
    