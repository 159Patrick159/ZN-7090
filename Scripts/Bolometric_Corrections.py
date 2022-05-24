def Martinez_cooling(Bmags,Vmags,BmagsErr,VmagsErr):
    
    '''Returns the bolometric magntiudes, their respective uncertainties and the BC given Martinez et al.
    bolometric corrections for a SNe in its cooling phase. Its input must be two arrays, B and V magnitudes
    respectively and their uncertainties. '''
    
    from uncertainties import ufloat,unumpy
    
    # Initialize the coefficients for the BC with respective uncertainties
    sigma = 0.12
    c0, c1, c2, c3, c4 = ufloat(-0.740,sigma), ufloat(4.472,sigma), ufloat(-9.637,sigma),\
    ufloat(9.075,sigma), ufloat(-3.290,sigma)
    # Define BC with coefficients
    BC  = lambda x: c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
    # Turn magntiude arrays into desireable data type for uncertainy handling
    Bmags = unumpy.uarray(Bmags,BmagsErr)
    Vmags = unumpy.uarray(Vmags,VmagsErr)
    color = Bmags - Vmags
    mbol = BC(color) + Bmags
    return(unumpy.nominal_values(mbol),unumpy.std_devs(mbol),BC(color),color)
