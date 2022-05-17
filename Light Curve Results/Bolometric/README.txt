Matthew Leung
April/May 2022

This directory contains bolometric light curves of SN KSP-ZN7090, which were constructed using the
method of bolometric corrections. See Lyman et al. (2013) and Martinez et al. (2021) for info
regarding bolometric corrections:
Lyman et al. (2013): https://doi.org/10.1093/mnras/stt2187
Martinez et al. (2021): https://doi.org/10.1051/0004-6361/202142075
The bolometric correction parameters in Lyman et al. (2013) were used here.
See Chapter 7 of my thesis.

Note that at the time of this work, the bolometric light curves were constructed with the
assumption that the redshift of KSP-ZN7090 was z=0.1.

------------------------------------------------------------------------------------------------

The bolometric light curves are saved as CSV files, which can be easily parsed with the Pandas
library in Python. The formatting for the filenames in this directory is:
<BC_type>.<LC_phase>.x-y.<num_MC_trials>.<offset>.<result_type>.<extension>
Where:
<BC_type> is either "Lyman" or "Martinez".
<LC_phase> is the phase of the light curve (e.g. radiative, cooling).
x and y are the bands used for the bolometric correction (see Section 7.2.1 of my thesis)
<num_MC_trials> is the number of Monte Carlo trials used in bolometric corrections
<offset> is whether or not an offset was used in the Monte Carlo trials; NO means no offset, O means offset
<result_type> is the type of the result (e.g. BC_VS_colour, BC_VS_time)

------------------------------------------------------------------------------------------------

Bol_LC_file_col_description.txt provides a description of the columns in the CSV light curve files.

Bol_LC_open_file.py contains example Python code to open the CSV files.
