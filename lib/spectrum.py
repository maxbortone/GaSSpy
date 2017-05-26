import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

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
            self.error = None
            self.ivar = None
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
            # determine flux error from inverse variance (ivar)
        else:
            # source is something else: look for minimum set of information
            if 'FLUX' in sp:
                self.flux = sp['FLUX']
            if 'IVAR' in sp:
                self.ivar = sp['IVAR']
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
        self.error_norm = None
        self.flux_interp = None
        self.error_interp = None
        self.lam_interp = None
        self.mask = np.isnan(self.error)
        self.SNR = 0
        self.S = 0
        self.N = 0

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
            # add spectral data
            for key in data_cols:
                sp[key.lower()] = data[key]
            # initialize error array if ivar is present
            if 'ivar' in sp:
                ivar = np.array(sp['ivar'])
                np.place(ivar, ivar==0, np.nan)
                sp['error'] = 1.0 / np.sqrt(ivar)
            sp['filename'] = filename
            f.close()
            return sp

    """
    Computes the signal-to-noise ratio of the spectrum

    INPUT:
        wlr: list or array of wavelength regions where to compute the SNR
        flag: boolean, if True outputs SNR value in each wavelength region
    """
    def signaltonoise(self, wlr, flag=False):
        # TODO: assert that spectrum has been deredshifted and normalized
        m = wlr.shape[0]
        SNRs = np.empty(m)
        S = np.empty(m)
        N = np.empty(m)
        for i in range(m):
            wave = 10**self.loglam_dered
            idx = (wave >= wlr[i][0]) & (wave <= wlr[i][1])
            s = np.median(self.flux_norm[idx])
            # use np.nanmedian instead of median since self.err == nan where self.ivar == 0
            if np.isnan(self.error_norm[idx]).all():
                print "Warning: all NaN error in wavelength range {}-{}!".format(wlr[i][0], wlr[i][1])
                n = np.nan
            else:
                n = np.nanmedian(self.error_norm[idx])
            SNRs[i] = s / n
            S[i] = s
            N[i] = n
        self.S = np.median(S)
        self.N = np.nanmedian(N)
        self.SNR = self.S / self.N
        if flag:
            return SNRs

    """
    De-redshift spectrum by redshift estimate @self.z

    NOTE:
        the log10 of 1+z is subtracted from the log10 wavelength in @self.loglam
    """
    def deredshift(self):
        self.loglam_dered = self.loglam - np.log10(1+self.z)

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
        self.error_norm = self.error / flux_median

    """
    Mask wavelengths in @wl with width @dw to remove contributions from sky
    emission lines, e.g. around 5577, 6300 and 6363 angstrom in rest frame

    INPUT:
        wl: array of wavelengths to mask
        dw: emission line width to be masked
    """
    def mask_skylines(self, wl, dw):
        # TODO: assert len(self.mask) == len(self.lam_interp)
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
        self.mask = np.ma.mask_or(self.mask, ms)

    def mask_sigma_peaks(self, sigma=5):
        sc = sigma_clip(self.flux_interp, sigma=sigma, iters=1)
        ms = sc.mask
        pk = np.flatnonzero(ms)
        for p in pk:
            if self.flux_interp[p] < 2:
                ms[p] = False
        self.mask = np.ma.mask_or(self.mask, ms)

    """
    Interpolate flux and noise on a grid of equally spaced wavelengths
    between the smallest and largest integer wavelength in self.loglam_dered

    INPUT:
        gs: spacing of wavelength grid, e.g. 1 angstrom
    """
    def interpolate(self, gs):
        # TODO: add ability to choose between flux and flux_norm
        if self.flux_norm is None or self.error_norm is None:
            self.normalize()
        # TODO: add ability to choose between loglam and loglam_dered
        # setup grid
        lam = 10**self.loglam_dered
        a = np.ceil(lam[0])
        b = np.ceil(lam[-1])
        lam_interp = np.arange(a, b, gs)
        # interpolate flux
        self.flux_interp = np.interp(lam_interp, lam, self.flux_norm)
        # TODO: check what happens if nan is at the boundary
        # interpolate error replacing nan by linear interpolation
        not_nan = np.logical_not(np.isnan(self.error_norm))
        self.error_interp = np.interp(lam_interp, lam[not_nan], self.error_norm[not_nan])
        self.mask = np.interp(lam_interp, lam, self.mask).astype(bool)
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
        # select flux, noise, wavelength and mask in the range
        self.flux_interp = self.flux_interp[idx]
        self.error_interp = self.error_interp[idx]
        self.lam_interp = self.lam_interp[idx]
        self.mask = self.mask[idx]
