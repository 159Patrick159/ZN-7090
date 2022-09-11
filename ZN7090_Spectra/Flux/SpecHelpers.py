import numpy as np

def spec_fits(filename, get_err=False, get_meta=False):
    from specutils.io import read_fits
    import astropy.units as u
    from astropy.io import fits
    from synphot import SourceSpectrum
    from synphot.models import Empirical1D

    #load fits file
    spec_fits = read_fits.read_fits_spectrum1d(filename)
    #read header info
    spec_meta = fits.getheader(filename)
    #check for multispec
    if isinstance(spec_fits, list):
        spec = spec_fits[1]
        if get_err:
            spec_err = spec_fits[3]
        print (spec_meta['SITEID'])
    else:
        spec = spec_fits
    #spectrum flux data
    flux = spec.data
    if spec_meta['BUNIT']=='erg/cm2/s/A':
        flux = flux * u.erg/u.cm**2/u.s/u.AA
        if get_err:
            flux_err = spec_err.data
            flux_err = flux_err * u.erg/u.cm**2/u.s/u.AA
    wave = spec.dispersion
    if isinstance(wave, u.Quantity):
        wave = wave.value * u.AA
    #create synphot spectrum object
    spec = SourceSpectrum(Empirical1D, points=wave,
                          lookup_table=flux, keep_neg=True)
    if get_err:
        spec_err = SourceSpectrum(Empirical1D, points=wave,
                          lookup_table=flux_err, keep_neg=True)
    #check if metadata is needed
    if get_meta:
        from astropy.coordinates import SkyCoord
        
        #calculate time observed
        datestr = 'DATE-OBS'
        if 'T' in spec_meta[datestr]:
            isot_obs = spec_meta[datestr]
        elif 'UT' in spec_meta:
            isot_obs = spec_meta[datestr]+'T'+spec_meta['UT']
        else:
            isot_obs = spec_meta[datestr]+'T'+spec_meta['UT-TIME']
        #exposure time
        exp_obs = spec_meta['EXPTIME']
        #RA, DEC observed
        RA, DEC = spec_meta['RA'], spec_meta['DEC']
        coord = SkyCoord(RA, DEC, unit=(u.hourangle, u.deg))
        RA, DEC = coord.ra.degree, coord.dec.degree
        #return spectrum and metadata
        #return spectrum
        if get_err:
            return (spec, spec_err, isot_obs, exp_obs, (RA,DEC))
        else:
            return (spec, isot_obs, exp_obs, (RA,DEC))
    else:
        #return spectrum
        if get_err:
            return (spec, spec_err)
        else:
            return (spec)

#SpecData = spec_fits("SFinalSpec.fits")
import specutils
#print(sys.prefix)
