#Matthew Leung
#March/May 2022
"""
The function load_LC_file is used to load the CSV light curve file of
KSP-ZN7090 into a pandas.DataFrame.

This example code plots the V band light curve.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_LC_file(filename):
    types = {'date':str, 'date[MJD]':np.float64, 'telescope':str, 'band':str,
             'M_0':np.float64, 'M_0_err':np.float64, 'M_0_lim':np.float64,
             'M_C':np.float64, 'M_C_err':np.float64, 'M_C_lim':np.float64,
             'M_E':np.float64, 'M_E_err':np.float64, 'M_E_lim':np.float64,
             'binned':str, 'binned_images':str}
    df = pd.read_csv(filename, delimiter=',', dtype=types, parse_dates=['date'], index_col=['date'])
    return df

def main():
    filename = "KSP-ZN7090_LC_20220309_correct.csv"
    df = load_LC_file(filename) #load the light curve file
    
    #Get the V band data
    V_band_df = df[df['band'] == 'V']
    V_band_times = V_band_df['date[MJD]'].to_numpy() #V band dates
    V_band_mag = V_band_df['M_E'].to_numpy() #V band magnitudes
    V_band_mag_err = V_band_df['M_E_err'].to_numpy() #V band magnitude uncertainties
    
    #Plot V band light curve
    plt.figure()
    plt.errorbar(V_band_times, V_band_mag, yerr=V_band_mag_err, fmt='o',
                 ecolor='tab:green', elinewidth=1, color='tab:green',
                 capsize=1, markeredgewidth=1, markeredgecolor='k')
    plt.xlabel('Date (MJD)')
    plt.ylabel('$V$ band Apparent Magnitude')
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    main()
    