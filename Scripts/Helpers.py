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
        delta_BC = abs(BC_Change) - abs(BC_Original)
        per_BC = delta_BC/BC_Original
        return(abs(per_BC/dc))
    
    if len(cs) == 5:
        # We are looking at a 4th order polynomail
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

def HO_PolyFit(dates,mags,mags_err,N):
    
    '''Performs a non-linear least square fit on a polynomial
    of degree 7 for a given light curve. It return the upper
    lower and mean fit parameteres for given fit.'''
    
    from scipy.optimize import curve_fit
    import numpy as np
    # Define high order polynomial 
    def polynomial_7(x,c0,c1,c2,c3,c4,c5,c6,c7):
        return(c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6 + c7*x**7)

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

    # Fit the rows of data
    coeffs=[]
    for row in big_data:
        popt,pcov = curve_fit(polynomial_7,dates,row)
        coeffs.append(popt)
    mean_coeffs = np.mean(coeffs,axis=0)
    
    return(mean_coeffs,coeffs)