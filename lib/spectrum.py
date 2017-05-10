import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class Spectrum:
    """
    Initialize a spectrum

    INPUT:
        source:     object containing spectral information like flux and noise,
                    but also redshift and coordinates (RA, DEC);
                    can also be the filename of a fits file
    """
    def __init__(self, source):
        if source is None:
            # source is empty: initialize a blank spectrum
            self.flux = None
            self.noise = None
            self.loglam = None
            self.z = None
            self.z_err = None
            self.ra = None
            self.dec = None
        elif source.endswith('.fits'):
            # source is a fits file: copy attributes
            sp = self.read(source)
            for key in sp:
                self.__dict__[key] = sp[key]
        else:
            # source is something else: look for minimum set of information
            if 'FLUX' in sp:
                self.flux = sp['FLUX']
            if 'NOISE' in sp:
                self.noise = sp['NOISE']
            elif 'IVAR' in sp:
                self.noise = sp['IVAR']
            if 'LOGLAM' in sp:
                self.loglam = sp['LOGLAM']
            if 'Z' in sp:
                self.z = sp['Z']
            if 'Z_ERR' in sp:
                self.z_err = sp['Z_ERR']
            if 'PLUG_RA' in sp:
                self.ra = sp['PLUG_RA']
            elif 'RA' in sp:
                self.ra = sp['PLUG_RA']
            if 'PLUG_DEC' in sp:
                self.dec = sp['PLUG_DEC']
            elif 'DEC' in sp:
                self.dec = sp['PLUG_DEC']

        # initialize additional attributes
        self.loglam_dered = None
        self.flux_norm = None
        self.noise_norm = None
        self.flux_interp = None
        self.noise_interp = None
        self.lam_interp = None
        self.mask = None
        self.SN = None

    """
    Read and initialize a spectrum from a fits file
    (https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html)

    INPUT:
        filename:   string with path to a fits file,
                    e.g. ./spectra/spec-DR13-1012-52649-639.fits
    """
    def read(self, filename):
        if filename.endswith('.fits'):
            f = fits.open(filename)
        else:
            raise ValueError('{} is not a fits file.'.format(filename))

        try:
            header = f[0].header
            data = f[1].data
            meta = f[2].data
        except IndexError:
            # TODO: handle exception
            pass
        else:
            keys = header.keys()
            data_cols = data.columns.names
            meta_cols = meta.columns.names
            sp = {}
            # read right ascension and declination of object
            if 'PLUG_RA' and 'PLUG_DEC' in keys:
                sp['ra'] = header['PLUG_RA']
                sp['dec'] = header['PLUG_DEC']
            else:
                # TODO: handle exception
                pass
            # read best redshift and error of object
            if 'Z' and 'Z_ERR' in meta_cols:
                sp['z'] = meta['Z']
                sp['z_err'] = meta['Z_ERR']
            else:
                # TODO: handle exception
                pass
            # read signal-to-noise of object:
            if 'SN_MEDIAN_ALL' in meta_cols:
                sp['sn_median'] = meta['SN_MEDIAN_ALL']
            else:
                pass
            # add spectral data
            for key in data_cols:
                sp[key.lower()] = data[key]
            # rename ivar key into noise
            sp['noise'] = sp.pop('ivar')
            # TODO: keep both fluxes
            # try:
            #     sp['flux'] = sp.pop('flux_dustcorr')
            # except:
            #     pass
            # add filename
            sp['filename'] = filename
            f.close()
            return sp

    """
    Computes the signal-to-noise ratio of the spectrum

    INPUT:
        wr: tuple or array of two floats representing a wavelength range,
            e.g. [4100, 4700]
    """
    def signaltonoise(self, wr=[4500, 4700], flag=False):
        # wave = 10**self.loglam_dered
        wave = self.lam_interp
        idx = (wave >= wr[0]) & (wave <= wr[1])
        S = self.flux_interp[idx]
        # N = self.noise_norm[idx].std()
        # N = self.noise_interp[idx]
        N = 1.0/np.sqrt(self.noise_interp[idx])
        w = self.noise_interp[idx].mean()
        if flag:
            return np.median(S), np.median(N), w
        else:
            return np.median(S) / np.std(S)
            # return np.median(S) / np.median(N)

    """
    De-redshift spectrum using the redshift information
    the fits file

    INPUT:
        rs:     (optional) float representing a redshift estimation,
                if set, rs will be used, otherwise self.z if it is defined
    """
    def deredshift(self, rs=None):
        if rs is not None:
            self.loglam_dered = self.loglam - np.log10(1+rs)
        elif self.z is not None:
            self.loglam_dered = self.loglam - np.log10(1+self.z)
        else:
            # TODO: handle exception
            print('No redshift defined')

    """
    Normalize flux and noise by the mean flux over a wavelength interval [a, b]

    INPUT:
        wr: tuple or array of two floats representing a wavelength range,
            e.g. [4100, 4700]
        dc: wheter to use the dust corrected flux or not
    """
    def normalize(self, wr=[4100, 4700], dc=False):
        # TODO: add ability to choose between loglam and loglam_dered
        if self.loglam_dered is None:
            self.deredshift()
        # setup mask around wavelength range to calculate mean flux
        wave = 10**self.loglam_dered
        idx = (wave >= wr[0]) & (wave <= wr[1])
        if dc:
            flux_median = np.median(self.flux_dustcorr[idx])
        else:
            flux_median = np.median(self.flux[idx])
        self.flux_norm = self.flux / flux_median
        self.noise_norm = self.noise / flux_median

    """
    Mask wavelengths to remove contributions from sky emission lines,
    for example around 5577, 6300 and 6363 angstrom in rest frame

    INPUT:
        wl: array of wavelengths to mask
        dw: emission line width to be masked
    """
    def setmask(self, wl, dw):
        # convert wavelengths into redshift frame
        wl = wl / (1+self.z)
        dw = dw / (1+self.z)
        idx = []
        # get masks for all intervalls
        for w in wl:
            a = w-dw
            b = w+dw
            idx.append(np.ma.masked_inside(self.lam_interp, a, b))
        # merge all masks together
        ms = np.ma.mask_or(idx[0].mask, idx[1].mask)
        for i in range(len(idx)-2):
            ms = np.ma.mask_or(ms, idx[i+2].mask)
        self.mask = ms

    """
    Interpolate flux and noise on a grid of equally spaced wavelengths
    between the smallest and largest integer wavelength in self.loglam_dered

    INPUT:
        gs: spacing of wavelength grid, e.g. 1 angstrom
    """
    def interpolate(self, gs):
        # TODO: add ability to choose between flux and flux_norm
        if self.flux_norm is None or self.noise_norm is None:
            self.normalize()
        # TODO: add ability to choose between loglam and loglam_dered
        # setup grid
        lam = 10**self.loglam_dered
        a = np.ceil(lam[0])
        b = np.ceil(lam[-1])
        lam_interp = np.arange(a, b, gs)
        # interpolate
        self.flux_interp = np.interp(lam_interp, lam, self.flux_norm)
        self.noise_interp = np.interp(lam_interp, lam, self.noise_norm)
        self.lam_interp = lam_interp

    """
    Rebase interpolated spectrum to the wavelenght range [w1, w2]

    INPUT:
        w1: left wavelength limit
        w2: right wavelength limit
    """
    def rebase(self, w1, w2):
        # TODO: print out a warning if [w1, w2] is not all in self.lam_interp
        # get indices of wavelength in the range [w1, w2]
        idx = (self.lam_interp >= w1) & (self.lam_interp <= w2)
        # select flux, noise and wavelength in the range
        # TODO: check if it is pointing to a new array
        self.flux_interp = self.flux_interp[idx]
        self.noise_interp = self.noise_interp[idx]
        self.lam_interp = self.lam_interp[idx]
        # select mask
        if self.mask is not None:
            self.mask = self.mask[idx]
