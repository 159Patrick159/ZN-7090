import numpy as np
def Uncertainties_BV_Martinez(Bmags,Vmags,BmagsErr,VmagsErr):
    
    """ Using the generalized uncertainty propagation formula we propagate the uncertainties
    for the bolometric magntiude for B-V correction using Martinez et al. coefficient and polynomial"""
    
    # Initialize the coefficients for the correction
    c0,c1,c2,c3,c4 = 0.740, 4.472, 9.637, 9.075, 3.290
    
    # Compute the colors
    colors = Bmags-Vmags
    
    # Compute color error
    color_err = np.sqrt(BmagsErr**2 + VmagsErr**2)
    
    # Compute bolometric correction error
    correction_err = ((c1) - 2*c2*colors + 3*c3*colors**2 - 4*c4*colors**3)*color_err
    
    # Compute bolometric magnitude error
    mbol_err = np.sqrt(correction_err**2 + BmagsErr**2)
    
    return(mbol_err)