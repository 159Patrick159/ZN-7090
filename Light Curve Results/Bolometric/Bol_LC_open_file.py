#Matthew Leung
#April/May 2022
"""
The function load_bol_luminosity_LC_file is used to load the CSV bolometric
light curve file of KSP-ZN7090 into a pandas.DataFrame.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_bol_luminosity_LC_file(filename):
    types = {'date':str, 'date[MJD]':np.float64, 
             'm_bol':np.float64, 'm_bol_err':np.float64,
             'M_bol':np.float64, 'M_bol_err':np.float64, 
             'L_cgs':np.float64, 'L_cgs_err':np.float64,
             'L_SI':np.float64, 'L_SI_err':np.float64}
    df = pd.read_csv(filename, delimiter=',', dtype=types, parse_dates=['date'], index_col=['date'])
    return df

def main():
    filename = "Lyman.radiative.B-V.1000000.NO.luminosity.csv"
    df = load_bol_luminosity_LC_file(filename) #load the light curve file
    
    #Get the bolometric luminosity in CGS units, assuming z=0.1
    L_cgs = df['L_cgs'].to_numpy()
    L_cgs_err = df['L_cgs_err'].to_numpy()
    times = df['date[MJD]'].to_numpy()

    #Plot the bolometric luminosity light curve
    plt.figure()
    plt.errorbar(times, L_cgs, yerr=L_cgs_err, fmt='o', color='dimgrey', elinewidth=1, capsize=1, markeredgewidth=1, markeredgecolor='k')
    plt.xlabel('Date (MJD)')
    plt.ylabel('Luminosity [erg/s]')
    plt.show()

if __name__ == "__main__":
    main()
