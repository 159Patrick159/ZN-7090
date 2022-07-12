# Create a class that keeps track of a specific date and the corresponding residual
class Date:
    def __init__(self, Day, Res):
        self.d = Day
        self.r = Res

# We first must identify all the interpolated points so we must look out for similar dates
def spot_interpolation(dates1,dates3):
    
    '''Returns the indices of the dates where there was interpolation
    dates1 is the original data, dates2 is the data we want dates1 to look like
    and dates3 is the interpolation of dates1'''
    
    import copy
    import math
    import numpy as np
    
    # Check if dates array are of correct sizes
    N1, N3 = len(dates1), len(dates3)
    if N1 > N3:
        raise Exception("Wrong length of original data")
    
    # This algorithm will match the dates from date3 to date1 by minimizing the residuals between the dates
    # because N1 < N3 there will be some left-over dates in N3 after all residual have been minimized
    # This left over dates are the interpolated dates!
    
    # Initialize a queue that 
    queue = copy.deepcopy(dates1)
    
    # Initialize array fro matched dates
    matches = []
    m_res = []
    
    # Check if queue is non-empty
    while len(queue) != 0:
        # Dequeue the first element of the queue
        day = queue[0]
        queue = np.delete(queue,0)
        
        # Minimize Residual
        best_res = math.inf 
        best_date = Date(0,best_res)
        
        for i in range(N3):
            current_res = abs(day-dates3[i])
            if current_res < best_res:
                best_date.d = dates3[i]
                best_date.r = current_res
                best_res = current_res
                
        # Once for loop has ended we have date that has the lowest residual
        matches.append(best_date.d)
        m_res.append(best_date.r)
        
    # We will have some repeats so we will eliminate those with higher residuals
    I = []
    for i in range(len(matches)):
        for j in range(len(matches)):
            if i != j:
                # Check for repeats
                if matches[i] ==  matches[j]:
                    
                    # Eliminate the repeat with higher residual
                    if m_res[i] > m_res[j]:
                        I.append(i)
                            
                    elif m_res[i] < m_res[j]:
                        I.append(j)
    
    matches = np.delete(matches,I)
    m_res = np.delete(matches,I)
    
    # We convert the dates arrays into sets
    matches_set = set(matches)
    dates3_set = set(dates3)
    # Set operation allows us to isolate for interpolated dates
    inter_dates = dates3_set - dates3_set.intersection(matches_set)
    return(inter_dates)

def Probe_sensitivity(colors,dc,cs):
    
    '''Probes for the sensitivity of a bolometric correction due to color,
    cs is an array where the coefficients of the correction are stored
    in order of highest polynomial'''
    
    # Check for length of polynomial
    if len(cs) == 3:
        # We are looking at a 2nd order polynomial
        c0,c1,c2 = cs
        BC = lambda x: c0 + c1*x + c2*x**2
        
        # Compute colors and their respecive changes
        BC_Original = BC(colors)
        BC_Change = BC(colors + colors*dc)
        delta_BC = (BC_Change - BC_Original)
        delta_C = (colors*dc - colors)
        per_BC = delta_BC/delta_C
        return(per_BC/dc)
    
    if len(cs) == 5:
        # We are looking at a 4th order polynomial
        c0, c1, c2, c3, c4 = cs
        BC = lambda x: c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
        
        BC_Original = BC(colors)
        BC_Change = BC(colors + colors*dc)
        delta_BC = (BC_Change - BC_Original)
        delta_C = (colors*dc - colors)
        per_BC = delta_BC/delta_C
        return(per_BC/dc)
    
def Remove_Data(dates,data,data_err,err_threshold,data_range):
    
    '''Removes data points that are outside the range and have
    an uncertainty greater than the threshold. Data range is a list
    of length 2 where the first entry is the lower bound and second
    entry is the upper bound'''
    
    # Select bounds
    upper_bound = data_range[1]
    lower_bound = data_range[0]
    
    # Initialize empty array
    good_data = []
    
    
    for i,val in enumerate(data):
        # Check if data is within range
        if val <= upper_bound and val >= lower_bound:
            # Check if data point has an uncertainty greater than limit
            if data_err[i] < err_threshold:
                good_data.append(i)
    
    # Slice original data and dates to only contain points 
    # that satisfy conditions
    dates = dates[i]
    data = data[i]
    data_err = data_err[i]
    
    return(dates,data,data_err)

def HO_PolyFit(dates,mags,mags_err,N,deg):
    
    '''Performs a non-linear least square fit on a polynomial
    of degree 6 or 7 through a montecarlo simulation for sampling 
    on the data points. Returns the averaged coeff and the coeff
    of each monte carlo simulation'''
    
    from scipy.optimize import curve_fit
    import numpy as np
    
    # Create array for sampled data
    big_data = []
    for i in range(len(mags)):
        # Sample N points from Gaussian distribution
        mag_s = np.random.normal(loc=mags[i],scale=mags_err[i],size=N)
        big_data.append(mag_s)

    # Every row represents a sample froma single data point.
    big_data = np.array(big_data)

    # We transprose the matrix to get every row to represent the sampled light curve
    big_data = big_data.T
    
    #Check for degree polynomial
    if deg == 6:
        # Define high order polynomial 
        def polynomial_6(x,c0,c1,c2,c3,c4,c5,c6):
            return(c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6)

        # Fit the rows of data
        coeffs=[]
        for row in big_data:
            popt,pcov = curve_fit(polynomial_6,dates,row)
            coeffs.append(popt)
        mean_coeffs = np.mean(coeffs,axis=0)
        return(mean_coeffs,coeffs)
    
    if deg == 7:
        # Define high order polynomial 
        def polynomial_7(x,c0,c1,c2,c3,c4,c5,c6,c7):
            return(c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6 + c7*x**7)

        # Fit the rows of data
        coeffs=[]
        for row in big_data:
            popt,pcov = curve_fit(polynomial_7,dates,row)
            coeffs.append(popt)
        mean_coeffs = np.mean(coeffs,axis=0)
        return(mean_coeffs,coeffs)

def Match_Lengths(d1,d2):
    import numpy as np
    '''Appends NaN to smallter lenght data set such that both arrays
    have equal lengths. If both arrays have same length then both arrays
    are returned untouched.'''
    
    l1,l2 = len(d1),len(d2)
    
    if l1 > l2:
        #d2 = np.append(d2, np.repeat(np.nan, l1-l2))
        d2 = np.append(d2, np.repeat(None, l1-l2))
    else:
        #d1 = np.append(d1,np.repeat(np.nan, l2-l1))
        d1 = np.append(d1,np.repeat(None, l2-l1))
        
    return(d1,d2)
    
def combine_data(d1,d2,d3):
    
    '''Combine 3 data set into a single array where index corresponds to their band 0 for B,
    1 for V and 2 for I. If arrays are different lengths then arrays are filled in with NaN'''
    
    # Ensure all arrays are of the same size
    d1, d2 = Match_Lengths(d1,d2)
    d1, d3 = Match_Lengths(d1,d3)
    d2, d3 = Match_Lengths(d2,d3) 
    
    result = []
    
    # Append all data into lists
    result.append(d1)
    result.append(d2)
    result.append(d3)
    return(result)

def MC_Simulation_Graph(mag1,mag2,mag1_err,mag2_err,c0,c1,c2,Dl,index):
    """
    Simulate a Gaussian sampling over the specified magnitudes and uncertainties
    and produces three historgram otlining the disitribuion of the apparet, absolute
    magnitudes and the bolometric luminosity. Additionally it prints the Pearson's
    Median Skeness of the bolometric luminosity distribution. Note: the luminosity 
    distance needs to be in units of parsecs.
    """
    # Import needed libraries
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define Bolometric correction given coefficients
    BC = lambda x: c0 + c1*x + c2*x**2
    
    # Define physical parameters of sun
    Lsun = 3.845e33
    Msun = 4.74
    
    fig = plt.figure(figsize=(15,5))
    a0 = plt.subplot(1,3,1)
    a1 = plt.subplot(1,3,2)
    a2 = plt.subplot(1,3,3)
    
    for i in range(index,index+1):
        s_mag1 = np.random.normal(loc=mag1[i],scale=mag1_err[i],size=1000)
        s_mag2 = np.random.normal(loc=mag2[i],scale=mag2_err[i],size=1000)

        s_color = s_mag1-s_mag2
        s_BC = BC(s_color)
        s_mbol = s_BC + s_mag1
        s_Mbol = s_mbol - np.log10((Dl/10))
        s_Lbol = (Lsun*10**(0.4*(Msun - s_Mbol)))

        # Calculate statistical significant parameters
        mbol_lq = np.percentile(s_mbol,25)
        mbol_uq = np.percentile(s_mbol,75)

        Mbol_lq = np.percentile(s_Mbol,25)
        Mbol_uq = np.percentile(s_Mbol,75)

        Lbol_lq = np.percentile(s_Lbol,25)
        Lbol_uq = np.percentile(s_Lbol,75)
        Lbol_med = np.median(s_Lbol)
        Lbol_mean = np.mean(s_Lbol)
        Lbol_std = np.std(s_Lbol)
        
        a0.hist(s_mbol, bins = 'fd',density=True, color = 'k', histtype = 'step', alpha = 0.5)
        a0.plot(s_mbol, np.full_like(s_mbol, -0.01), '|k', markeredgewidth = 1, alpha = 0.2)
        a0.axvline(x = mbol_lq, color = 'k', label = '50% CI', linestyle = '--')
        a0.axvline(x = mbol_uq, color = 'k', linestyle = '--')
        a0.set_xlabel("Bolometric Apparent Mag",fontsize=14)
        a0.set_ylabel("Probability Density",fontsize=14)
        a0.legend(prop={'size':9})

        a1.hist(s_Mbol, bins = 'fd',density=True, color = 'k', histtype = 'step', alpha = 0.5)
        a1.plot(s_Mbol, np.full_like(s_Mbol, -0.01), '|k', markeredgewidth = 1, alpha = 0.2)
        a1.axvline(x = Mbol_lq, color = 'k', label = '50% CI', linestyle = '--')
        a1.axvline(x = Mbol_uq, color = 'k', linestyle = '--')
        a1.set_xlabel("Bolometric Absolute Mag",fontsize=14)
        a1.set_ylabel("Probability Density",fontsize=14)
        a1.legend(prop={'size':9})
                                   
        a2.hist(s_Lbol, bins = 'fd',density=True, color = 'k', histtype = 'step', alpha = 0.5)
        a2.axvline(x = Lbol_lq, color = 'k', label = '50% CI', linestyle = '--')
        a2.axvline(x = Lbol_uq, color = 'k', linestyle = '--')
        a2.axvline(x = Lbol_med, color = 'r', label = 'Median', linestyle = '-',linewidth=0.9)
        a2.axvline(x = Lbol_mean,color='orange',label='Mean',linewidth=0.9)
        a2.axvline(x = Lbol_mean + Lbol_std,color='b',label='Std',alpha=0.5,ls="-.")
        a2.plot(s_Lbol, np.full_like(s_Lbol, 0.17e-31), '|k', markeredgewidth = 1, alpha = 0.1)
        a2.axvline(x = Lbol_mean - Lbol_std,color='b',alpha=0.5,ls="-.")
        a2.set_xlabel("Bolometric Luminosity",fontsize=14)
        a2.set_ylabel("Probability Density",fontsize=14)
        a2.legend(prop={'size':9}) 
        plt.tight_layout()
        plt.show()
        print("Pearson's Median Skewness of Bolometric Luminosity Distribution")
        print(3*(Lbol_mean - Lbol_med)/Lbol_std)

def MC_Simulation_Array(mag1,mag2,mag1_err,mag2_err,c0,c1,c2,Dl,index):
    """
    Performs Gaussian sampling on specified magnitudes to compute the bolometric apparent, absolute
    magnitudes and luminosity. Function returns the correspond arrays for the specified data point
    from index parameter.
    """
    import numpy as np
    
    # Define Bolometric correction given coefficients
    BC = lambda x: c0 + c1*x + c2*x**2
    
    # Define physical parameters of sun
    Lsun = 3.845e33
    Msun = 4.74
    
    s_mag1 = np.random.normal(loc=mag1[index],scale=mag1_err[index],size=1000)
    s_mag2 = np.random.normal(loc=mag2[index],scale=mag2_err[index],size=1000)

    s_color = s_mag1-s_mag2
    s_BC = BC(s_color)
    s_mbol = s_BC + s_mag1
    s_Mbol = s_mbol - np.log10((Dl/10))
    s_Lbol = (Lsun*10**(0.4*(Msun - s_Mbol)))

    return(s_mbol,s_Mbol,s_Lbol)