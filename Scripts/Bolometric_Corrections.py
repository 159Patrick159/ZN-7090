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

def Layman1(mag1,mag2,mag1err,mag2err,range_eff,c0,c1,c2,rms,dates):
    
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
        if color < upper and color > lower:
            good_mags.append(i)
            
    # Only select the pair mags that are within the effective range of color
    mag1 = mag1[good_mags]
    mag2 = mag2[good_mags]
    
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
    
def Layman2(mag1,mag2,mag1err,mag2err,range_eff,c0,c1,c2,rms,dates):
    
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
        if color < upper and color > lower:
            good_mags.append(i)
            
    # Only select the pair mags that are within the effective range of color
    #mag1 = mag1[good_mags]
    #mag2 = mag2[good_mags]
    
    # And the corresponding dates
    #dates = dates[good_mags]
    
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
    