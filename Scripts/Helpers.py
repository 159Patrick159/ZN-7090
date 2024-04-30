# Import needed libraries
import numpy as np
import copy
import math

#Create a class that keeps track of a specific date and the corresponding residual
class Date:
    def __init__(self, Day, Res):
        self.d = Day
        self.r = Res

# We first must identify all the interpolated points so we must look out for similar dates
def spot_interpolation(dates1,dates3):
    
    '''Returns the indices of the dates where there was interpolation
    dates1 is the original data, dates2 is the data we want dates1 to look like
    and dates3 is the interpolation of dates1
    '''
    
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

def Mag2Luminosity(mag,mag_err,Dl):
    """
    Converts observed magntiudes to luminosities by applying the luminosity distance
    and the distance modulus equation. Dl needs to be in units of pc
    """
    
    # Initialize needed constants
    Lsun = 3.845e33 
    Msun = 4.74
    
    mtop = mag+mag_err
    
    # Convert to absolute magnitude
    M = mag - 5*np.log10(Dl/(10))
    Mtop = mtop - 5*np.log10(Dl/(10))
    # Convert to luminosity
    L = (Lsun*100**((Msun - M)/5))
    Ltop = (Lsun*100**((Msun - Mtop)/5))
    
    return(L,L-Ltop)

def RabinakShockCooling(t5,E51,R13,M,k34,fp):
    # Define needed constants
    h = 6.6260755e-27 #erg*s
    c = 2.99792458e10 #cm/s
    
    # Compute Photospheric Temperature
    Tph = 1.6*(fp**(-0.037)*E51**0.027*R13**(1/4))/(M**0.054*k34**0.28) * t5**(-0.45)
    
    # Compute Bolometric Luminosity
    L = 8.5e42*(E51**0.92*R13)/(fp**0.27*M**0.84*k34**0.92)*t5**(-0.16)
    
    return(L,Tph)

#function: stefan Boltzmann's law
def SBlaw(T):
    sb = 5.67051e-5 #erg/s/cm2/K4
    #black body total flux
    integ = sb*np.power(T,4) #ergs/s/cm2
    return (integ)

def blackbod(x, T):
    # Change units of wavelength to cgs
    #wave = x*1e-8 #angstrom to cm
    wave = x
    
    # Define physical constants
    h = 6.6260755e-27 #erg*s
    c = 2.99792458e10 #cm/s
    k = 1.380658e-16 #erg/K
    freq = c/wave #Hz
    
    # Define spectral radiance from planck distribution
    p_rad = (2*h*freq**3/c**2)/(np.exp(h*freq/(k*T))-1.0) #erg/s/cm2/rad2/Hz power per area per solid angle per frequency
    
    # Integrat planck distribution over solid angle
    p_int = np.pi*p_rad #erg/s/cm2/Hz #power per area per frequency [fnu]
    return (p_int)

#function: normalized planck distribution (wavelength)
def planck(x, T):
    #black body total flux
    integ = SBlaw(T) #erg/s/cm2
    #blackbody distribution
    p_int = blackbod(x,T) #erg/s/cm2/Hz #power per area per frequency [fnu]
    #normalized planck distribution
    return (p_int/integ) #1/Hz, luminosity density

#function: L, T fluxes derived using blackbody spectrum
def BBflux(Lc,Teff,wave,z,dl):
    '''
    Return flux density per unit frequency given theoretical
    bolometric luminosity and temperature expressions.
    '''
    #give wave in observer frame
    
    #luminosity distance [pc -> cm]
    # dl = dl*3.086*10**18
    Area = 4.0*np.pi*np.square(dl) #cm^2
    #kasen model in observer band
    Lc_wave = planck(wave/(1.0+z),Teff)*Lc/Area
    #Lc_angle_wave = np.nan_to_num(Lc_angle_wave)
    #ergs/s/Hz/cm^2, luminosity density in observer frame
    # Lc is in erg/s so Lc_wave is in erg/s/cm2/Hz
    return (Lc_wave)
    
def chi_square(y_measured, y_expected,errors):
    return np.sum( np.power((y_measured - y_expected),2) / np.power(errors,2) )

def chi_square_reduced(y_measured,y_expected,errors,number_parameters):
    return chi_square(y_measured,y_expected,errors)/(len(y_measured) - number_parameters)

def Nice_Plots():
    import matplotlib.pyplot as plt
    style = 'default'
    tdir = 'in'

    major=5
    minor=3

    font = 'serif'

    plt.style.use(style)

    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir

    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor

    plt.rcParams['font.family'] = font

def day_mjd(day, year):
    ################################################################
    # Desc: Converts day of year float to mjd time.                #
    # ------------------------------------------------------------ #
    # Imports: astropy.time.(Time, TimeDelta)                      #
    # ------------------------------------------------------------ #
    # Input                                                        #
    # ------------------------------------------------------------ #
    #  day: float time in days since start of year YYYY            #
    # year: int reference year YYYY                                #
    # ------------------------------------------------------------ #
    # Output                                                       #
    # ------------------------------------------------------------ #
    # time: float time format in mjd                               #
    ################################################################
    from astropy.time import Time, TimeDelta
        
    #create astropy time object
    t_ref = str(year)+"-01-01T00:00:00.000"
    t_ref = Time(t_ref, format='isot', scale='utc')
    #create astropy time difference object
    t_diff = TimeDelta(day, format='jd')
    #return isot time
    return (t_ref+t_diff).mjd

def m2M(m,merr,z):
    import numpy as np
    import scipy.integrate as integrate
    import astropy.units as u
    from astropy.constants import c
    from uncertainties import unumpy,ufloat
    from uncertainties.umath import log

    # Define cosmological density parameters and H0 as mathew did
    OM = 0.27
    OK = 0.00
    OL = 0.73
    H0 = (74.2*u.km/u.s/u.Mpc).to(1/u.s)

    # # Parameters used by afsariard
    # H0 = (67.4*u.km/u.s/u.Mpc).to(1/u.s)
    # OM = 0.315
    # OK = 0.00
    # OL = 0.685

    # Numerically integrate for the comoving distance at z
    def func(z,M,K,L):
        return(1/np.sqrt(M*(1+z)**3 + K*(1+z)**2 + L))
    
    DC = c/H0 * integrate.quad(func,0,z,args=(OM,OK,OL))
    # Compute comovin distance
    DL = DC*(1+z)
    # Compute luminosity disance
    Dl, Dl_err = DL[0], DL[1]
    print("Distance to soruce from integration of z")
    print(Dl.to(u.Mpc),"Mpc (+/-)",Dl_err.to(u.Mpc))
    m_u = unumpy.uarray(m,merr)

    # Compute K-correction term
    K = -2.5*np.log10(1+z)

    M = m_u - 5*np.log10((Dl.to(u.pc)).value/((10*u.pc)).value) - K
    return(unumpy.nominal_values(M),unumpy.std_devs(M))

def FluxDensity(mx,mx_err,wave,fwave0=None,fnu0=None):
    '''
    Computes the flux density per unit wavelength for
    specified apparent magntiude and effective wavelength
    Zero point flux density per unit wavelnegth or 
    zero point flux density per unit frequency is must 
    be provided for conversion.
    '''
    from uncertainties import unumpy
    # Constants
    c = 2.99792458e10 #cm/s
    
    # Convert wavelenght from um to cm
    # wave = wave*1e-4
    # Wavelength is already in cm
    wave = wave.value
    # Initialize uncertainties arrays
    m = unumpy.uarray(mx,mx_err)
    
    # Check if flux density per unit wavelength is provided
    if fwave0 is not None:
        print("Computing flux densities per unit wavelength")
        f = fwave0*np.power(10,(-m/2.5)) # erg/s/cm2/A
        # Dae-sik error
        ferr = np.sqrt(0.921)*unumpy.nominal_values(f)*mx_err
    elif fnu0 is not None:
        # Convert from fnu to fwave
        print("Computing flux densities per sunit wavelegth")
        f = (c/wave**2)*fnu0*np.power(10,-m/2.5)

        # Convert from erg/s/cm2/cm -> erg/s/cm2/A
        f = f/1e8
        ferr = np.sqrt(0.921)*unumpy.nominal_values(f)*mx_err
    else:
        print("Must provide a zero point flux density")
    
    return(unumpy.nominal_values(f),unumpy.std_devs(f),ferr)
    # return(unumpy.nominal_values(f),ferr)

def BBPlank(x,T):  
    import astropy.units as u 
    h = 6.6260755e-27 *u.erg *u.s #erg*s
    c = 2.99792458e10 *u.cm / u.s #cm/s
    k = 1.380658e-16 * u.erg /u.K #erg/K
    term1 = 2*np.pi*h*c**2/x**5
    term2 = 1/(np.exp(h*c/x/k/T)-1)
    return term1 * term2

def FluxWave(L,T,wave,D,z):
    #########################################################################
    # Function returns the corresponding monochromatic flux density         #
    # per unit wavelenth (cgs) for given bolometric luminosity, temperature #
    # central wavelength, and distance to source, temperature is corrected  #
    # for redshift anc color according to Sapir & Waxman model              #
    #########################################################################
    # Inputs: L -> erg/s, T -> K, wave -> cm, D -> cm                       #
    #########################################################################
    # Output: Fwave -> ergs/s/cm2/cm                                        #
    #########################################################################
    import astropy.units as u
    sb = 5.67051e-5 * u.erg/u.s/u.cm**2/u.K**4 #erg/s/cm2/K4
    # Convert from Tph to Tcol
    Tcol = 1.1*T # for convective envelopes
    # Intrinsic temperature of blackbody
    Tz = Tcol/(1+z)
    term1 = L/4/np.pi/D**2/sb/Tz**4
    return(term1*BBPlank(wave,Tz))


def Fnu2Fwave(fnu,wave):
    # Convert flux densities per unit frequency
    # to flux densities per unit wavelength
    c = 2.99792458e10 #cm/s
    return (c/wave**2)*fnu

############################# Function for Simultanous Fitting #############################
def One_Model(t, R, vs, M, fp, w):
    '''Uses the extended model from sapir & waxman to get the 
    flux density for 'w' band'''
    from Helpers import FluxWave
    import astropy.units as u
    # Compute the early T and L from S&W
    kappa = 1 # Set kappa 0.34 cgs
    Trw = 1.61 * (vs**2*t**2/(fp*M*kappa))**0.027 *R**(1/4) / kappa**(1/4) *t**(-1/2) # eV
    Lrw = 2.0 * 10**42 *(vs*t**2/(fp*M*kappa))**(-0.086) * (vs**2*R)/kappa  # erg/s
    Trw = Trw*u.eV
    Lrw = Lrw*u.erg/u.s

    # We can also compute the ejecta energy
    # E = (vs*10**(8.5)/1.05/fp**(-2*0.191))**2*M

    # Calculate envelope mass from my derivation (check notes)
    Menv = fp**2*M/(fp**2+1)

    #####################################################################################
    # Apply supression factor to Lbol at later times
    # Compute transparency time tr
    # Comments: To compute the transparency time  we need to know the envelope mass,
    # this is a very hard paremter to get because it depends on the progenitors density
    # structure. However, from SW they provide bounds for such quatity 0.1< Mc/Menv< 10
    # and for convective effective envelopes we have that fp = sqrt(Mc/Menv), hence 
    # tr = 13 * fp**(0.191/2) * (Menv)**(3/4) * (M/Menv)**(1/4) * (E/10**(51))**(-1/4) # days
    #####################################################################################

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

    # Apply color factor fT=1.1
    Tc = 1.1*Tc*u.eV
    
    # Convert temperature from eV to K
    kB = 8.617333262*1e-5*u.eV/u.K #eV/K
    Tc = Tc/kB
    Trw = Trw/kB

    # Define redshift
    #z = 0.08739904944703399 # New analysis shows new redshift
    z = 0.08918

    # Define distance
    #Dl = 377.044*1e6 * u.pc # In case of using apparent magnitude
    Dl = 385.2*1e6 * u.pc #Updated luminosity distance
    # Dl = 10 * u.pc # If using absolute magnitudes

    Dl = Dl.to(u.cm) # Convert to cm
    # L: erg/s, T: K, w: cm, Dl: cm
    fpred = FluxWave(Lc,Tc,w,Dl,z) # My function returns erg/s/cm2/cm
    # fpred_sub = FluxWave(Lsup,Trw,w,Dl,z)
    # Convert to erg/s/cm2/A
    fpred = fpred.to(u.erg/u.s/u.cm**2/u.AA)
    # fpred_sub = fpred_sub.to(u.erg/u.s/u.cm**2/u.AA)
    #return(fpred_sub.value)
    return(fpred.value)