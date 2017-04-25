import os
import numpy as np
from scipy.optimize import curve_fit
from astroquery.sdss import SDSS
from astropy.io import fits
from ppxf import ppxf
import ppxf_util as util
import miles_util as lib
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
    Signal to noise analysis

    INPUT:
        wr: tuple or array of two floats representing a wavelength range,
            e.g. [4100, 4700]
    """
    # TODO: on which flux should the analysis be performed?
    def signaltonoise(self, wr=[4100, 4700]):
        wave = 10**self.loglam
        idx = (wave >= wr[0]) & (wave <= wr[1])
        S = self.flux[idx].mean()
        N = self.noise[idx].mean()
        self.SN = S / N

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
            flux_mean = self.flux_dustcorr[idx].mean()
        else:
            flux_mean = self.flux[idx].mean()
        self.flux_norm = self.flux / flux_mean
        self.noise_norm = self.noise / flux_mean

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
        self.flux_interp = self.flux_interp[idx]
        self.noise_interp = self.noise_interp[idx]
        self.lam_interp = self.lam_interp[idx]
        # select mask
        if self.mask is not None:
            self.mask = self.mask[idx]


class Stack:
    """
    Initialize stack with a list of spectra
    INPUT:
        filenames:  list of filenames of fits files or Spectrum class instances
        name:       name of the stack instance
    """
    def __init__(self, filenames, name=None):
        # TODO: rename filenames and rewrite test
        if isinstance(filenames[0], Spectrum):
            self.spectra = filenames
            self.N = len(filenames)
        elif isinstance(filenames[0], basestring):
            self.N = len(filenames)
            self.spectra = []
            for i in range(self.N):
                sp = Spectrum(filenames[i])
                self.spectra.append(sp)
        else:
            self.spectra = None
            self.N = 0
        if name:
            self.name = name
        # initialize attributes
        # wheter sky lines are being masked or not
        self.masking = None
        # length of stacked spectrum
        self.M = None
        # stores the weighted sum of the spectra at each wavelength pixel
        self.P = None
        # stores the sum of the weights of the spectra at each wavelength pixel
        self.S = None
        # wavelength pixels of stacked spectrum
        self.wave = None
        # flux of stacked spectrum
        # TODO: define a working flux that can be set to the stacked flux
        #  or the bias corrected one
        self.flux = None
        # noise of stacked spectrum
        self.noise = None
        # number of spectra that contributed to each wavelength pixel
        self.contributions = None
        # dispersion of the sample at each wavelength pixel
        self.dispersion = None
        # signal to noise ratio of the stacked spectrum
        self.SN = None
        # MILES template that is being used for the ppxf fit
        self.template = None
        # ppxf fit result
        self.pp = None
        # difference between stacked spectrum and ppxf bestfit
        self.residual = None
        # weights by which each template was multiplied to best fit
        # the stacked spectrum
        self.weights = None
        # mean logarithmic age
        self.mean_log_age = None
        # mean metallicity
        self.mean_metal = None
        # mass to light ratio assuming Salpeter IMF
        # TODO: check if this makes sense with temp='kb'
        self.mass_to_light = None

    """
    Calculate the weighted average of the stack

    INPUT:
        wr:     tuple or array of two floats representing a wavelength range,
                e.g. [4100, 4700]
        gs:     spacing of wavelength grid, e.g. 1 angstrom
        wl:     wavelengths of the sky spectrum to be masked
        dw:     width of region around wl to be masked
        dc:     wheter to use th edust corrected flux or not

    OUTPUT:
        lam:    wavelenghts of stacked spectrum
        flux:   flux of stacked spectrum
        noise:  noise of stacked spectrum
        contrs: contributions from spectra to flux at each wavelength
    """
    def average(self, wr, gs, wl=None, dw=None, dc=False):
        # set masking flag
        self.masking = (wl is not None) and (dw is not None)
        # prepare spectra
        for i in range(self.N):
            sp = self.spectra[i]
            sp.deredshift()
            sp.normalize(wr, dc)
            sp.interpolate(gs)
            if self.masking:
                sp.setmask(wl, dw)
        # find maximal common wavelength range limits
        w1 = self.spectra[0].lam_interp[0]
        w2 = self.spectra[0].lam_interp[-1]
        for i in range(self.N-1):
            v1 = self.spectra[i+1].lam_interp[0]
            v2 = self.spectra[i+1].lam_interp[-1]
            if v1 > w1:
                w1 = v1
            if v2 < w2:
                w2 = v2
        # rebase all spectra to wavelength range [w1, w2]
        for i in range(self.N):
            sp = self.spectra[i]
            sp.rebase(w1, w2)
        # setup output arrays
        self.M = int(w2-w1+1)
        lam = self.spectra[0].lam_interp
        flux = np.empty(self.M)
        noise = np.empty(self.M)
        contrs = np.empty(self.M)
        P = np.empty(self.M)
        S = np.empty(self.M)
        # stack spectra at each wavelength in the common range
        # according to mask of the spectra
        for i in range(self.M):
            p = 0
            s = 0
            n = 0
            c = self.N
            for j in range(self.N):
                sp = self.spectra[j]
                if self.masking and sp.mask[i]:
                    c -= 1
                else:
                    p += sp.flux_interp[i] * sp.noise_interp[i]
                    s += sp.noise_interp[i]
                    n += sp.noise_interp[i]**2
            if s == 0:
                flux[i] = 0
            else:
                P[i] = p
                S[i] = s
                flux[i] = p / s
            noise[i] = np.sqrt(n) / c
            contrs[i] = c
        self.wave = lam
        self.flux = flux
        self.noise = noise
        self.contributions = contrs
        self.P = P
        self.S = S
        return lam, flux, noise, contrs

    """
    Use the jackknife method to estimate the dispersion in the sample
    """
    def jackknife(self):
        # TODO: assert flux_interp and noise_interp are not None
        disp = np.empty(self.M)
        for i in range(self.M):
            p = self.P[i]
            s = self.S[i]
            g = 0
            for j in range(self.N):
                sp = self.spectra[j]
                if not (self.masking and sp.mask[i]):
                    n = sp.noise_interp[i]
                    f = sp.flux_interp[i]
                    g += ((p*(s-n) - s*(p-n*f)) / (s*(s-n)))**2
            c = self.contributions[i]
            disp[i] = g * (c - 1) / c
        self.dispersion = np.sqrt(disp)
        if self.dispersion.mean() > 0.01*self.flux.mean():
            # TODO: look into
            #   - 6560+-60 (H_alpha emission region)
            #   - continuum at 6800-7200
            # give out percentage difference
            if self.name:
                print("Sample dispersion of {} is high".format(self.name))
            else:
                print("Sample dispersion is high")
        return disp

    """
    Correct the stacked spectrum for possible sampling bias
    """
    def correct(self):
        # TODO: assert flux, flux_interp and noise_interp are not None
        corr = np.zeros(self.M)
        for i in range(self.M):
            p = self.P[i]
            s = self.S[i]
            for j in range(self.N):
                sp = self.spectra[j]
                if not (self.masking and sp.mask[i]):
                    n = sp.noise_interp[i]
                    f = sp.flux_interp[i]
                    corr[i] += (p - n*f) / (s - n)
            c = self.contributions[i]
            corr[i] = (c-1)*(corr[i]/c - self.flux[i])
        self.correction = corr
        self.flux -= self.correction
        return corr

    """
    Signal to noise analysis on stacked spectrum

    INPUT:
        wr: tuple or array of two floats representing a wavelength range,
            e.g. [4100, 4700]
    """
    def signaltonoise(self, wr):
        idx = (self.wave >= wr[0]) & (self.wave <= wr[1])
        S = self.flux[idx].mean()
        N = self.noise[idx].mean()
        self.SN = S / N

    """
    Fit the stacked spectrum using the Penalized Pixel Fitting method ppxf by
    Michele Cappellari (http://www-astro.physics.ox.ac.uk/~mxc/software/) with
    the Miles stellar library (http://www.iac.es/proyecto/miles/)

    INPUT:
        temp:   choose between 4 different templates with different IMF,
                [default, kb, ku, un]
    """
    def ppxffit(self, temp='default', refit=False):
        # CONSTANTS:
        # speed of light in km/s
        c = 299792.458
        # SDSS instrumental resolution
        FWHM_gal = 2.76
        # path to miles stellar library
        path = os.path.dirname(__file__).split('lib')[0] + 'miles_models/'
        templates = {
            'default': 'MILES_default/Mun1.30*.fits',
            'kb': 'MILES_Padova00_kb/Mkb1.30*.fits',
            'ku': 'MILES_Padova00_ku/Mku1.30*.fits',
            'un': 'MILES_Padova00_un/Mun1.30*.fits'
        }
        miles_path = path + templates[temp]
        self.template = temp
        # redshift is set to zero since stacked spectrum is calculated
        # in rest frame
        z = 0.0

        # run bias correction
        # TODO: cannot run bias correction if residuals in balmer lines have
        # been subtracted since self.P and self.S are not updated
        if not refit:
            self.correct()

        # bring galaxy spectrum to same wavelength range as the one of
        # the stellar library
        mask = (self.wave > 3540) & (self.wave < 7409)
        wave = self.wave[mask]
        flux = self.flux[mask]
        noise = self.noise[mask]
        # rebin flux logarithmically for ppxf
        flux, loglam, velscale = util.log_rebin([wave[0], wave[-1]], flux)
        flux = flux / np.median(flux)
        # initialize miles stellar library
        miles = lib.miles(miles_path, velscale, FWHM_gal)
        # determine goodpixels
        lam = np.exp(miles.log_lam_temp)
        dv = c*np.log(lam[0]/wave[0])  # km/s
        lam_range_temp = [lam.min(), lam.max()]
        goodpixels = util.determine_goodpixels(loglam, lam_range_temp, z,
                                               refit=refit)
        # initialize start values
        # TODO: check if initial estimate can be optimized to reduce
        # comp time, z should be set to mean redshift of stacked spectrum
        vel = c*np.log(1 + z)   # galaxy velocity in km/s
        start = [vel, 180.]     # starting guess for [V, sigma] in km/s
        # TODO: determine importance and effect of this parameter on
        # the quality of the fit
        delta = 0.004       # regularization error
        # fit stacked spectrum with ppxf
        # NOTES:
        #   - dv is necessary as the galaxy and the template do not have the
        #     same starting wavelength and therefore a velocity shift is
        #     applied to the template. All velocities are measured with respect
        #     to dv using the ppxf keyword vsyst
        #   - since we are not interested in the kinematics additive
        #     polynomials are excluded (degree=-1) and only multiplicative ones
        #     are used (mdegree=10), in order to preserve the line strenght of
        #     spectral features
        #   - the sigma clipping method is deactivated (clean=False), since we
        #     don't expect to have residual emissions or cosmic rays in our
        #     sample and we penalize these features during the stacking process
        pp = ppxf(miles.templates, flux, noise, velscale, start,
                  goodpixels=goodpixels, plot=False, moments=4, degree=-1,
                  vsyst=dv, clean=False, mdegree=10, regul=1./delta,
                  quiet=True)
        weights = pp.weights.reshape(miles.n_ages, miles.n_metal)/pp.weights.sum()
        self.mean_log_age, self.mean_metal = miles.mean_age_metal(weights, quiet=True)
        self.mass_to_light = miles.mass_to_light(weights, band="r", quiet=True)
        self.pp = pp
        self.pp.lam = np.exp(loglam)
        self.weights = weights
        self.residual = self.pp.galaxy - self.pp.bestfit

    """
    Subtract residuals in the Balmer lines from stacked spectrum
    and refit without masking the Balmer lines in goodpixel
    """
    def refit(self):
        # fit balmer lines in self.residual with a gaussian
        # to determine the width
        def gaus(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2.0*sigma**2))
        balmer_lines = [4101.76, 4340.47, 4861.33]  # Hdelta, Hgamma, Hbeta
        gaussians = []
        for line in balmer_lines:
            idx = (self.pp.lam >= np.round(line)-50) & (self.pp.lam <= np.round(line)+50)
            wv = self.pp.lam[idx]
            rs = self.residual[idx]
            popt, pcov = curve_fit(gaus, wv, rs, p0=[1.0, rs.mean(), rs.std()])
            print rs
            gs = gaus(wv, popt[0], popt[1], popt[2])
            gaussians.append(np.array([wv, gs]))
        return gaussians

    """
    Plot stacked spectrum, noise, dispersion and bias correction

    INPUT:
        filename:   plot filename
        title:      plot title
    """
    def plotStack(self, filename, title=None):
        mpl.rcParams['mathtext.fontset'] = 'stixsans'
        mpl.rcParams['font.family'] = 'sans'
        mpl.rcParams['font.serif'] = 'STIXGeneral'
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = [
               r'\usepackage{siunitx}',
               r'\sisetup{detect-all}',
               r'\usepackage{sans}',
               r'\usepackage{sansmath}',
               r'\sansmath',
               r'\DeclareSIUnit\erg{erg}'
        ]
        wv = self.wave
        f, axes = plt.subplots(2, 1, sharex=True, figsize=(16, 9))
        axes[0].plot(wv, self.flux, 'b', label="stacked flux")
        # axes[0].fill_between(wv, fl-ns, fl+ns, facecolor="blue", alpha="0.5")
        axes[0].plot(wv, self.noise, 'g', label="noise")
        axes[0].plot(wv, self.dispersion, 'r', label="dispersion")
        axes[0].plot(wv, self.flux-self.correction, 'y', label="bias corrected")
        axes[0].set_ylabel(r"$F_{\lambda}$ [\SI{e-17}{\erg\per\second\per\square\centi\meter\per\angstrom}]")
        axes[0].legend(loc="best", frameon=False)
        axes[1].plot(wv, self.contributions)
        axes[1].set_ylabel("number of spectra")
        axes[1].set_xlabel(r"$\lambda$ [\si{\angstrom}]")
        axes[1].set_ylim([self.contributions.min()-5, self.N+5])
        axes[1].yaxis.set_major_locator(MultipleLocator(5))
        if title is not None:
            axes[0].set_title(title)
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    """
    Plot stacked spectrum and the best fit obtained from ppxffit

    INPUT:
        filename:   plot filename
        title:      plot title
    """
    def plotFit(self, filename, title=None):
        mpl.rcParams['mathtext.fontset'] = 'stixsans'
        mpl.rcParams['font.family'] = 'sans'
        mpl.rcParams['font.serif'] = 'STIXGeneral'
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = [
               r'\usepackage{siunitx}',
               r'\sisetup{detect-all}',
               r'\usepackage{sans}',
               r'\usepackage{sansmath}',
               r'\sansmath',
               r'\DeclareSIUnit\erg{erg}'
        ]
        wv = self.pp.lam
        # ns = self.pp.noise
        f, axes = plt.subplots(1, 1, figsize=(16, 9))
        axes.plot(wv, self.pp.galaxy, 'b', label="stacked spectrum")
        # axes.fill_between(wv, fl-ns, fl+ns, facecolor="blue", alpha="0.5")
        axes.plot(wv, self.pp.bestfit, 'g', label="ppxf fit", alpha="0.5")
        axes.plot(wv, self.residual, 'r', label="residual")
        axes.set_ylabel(r"$F_{\lambda}$ [\SI{e-17}{\erg\per\second\per\square\centi\meter\per\angstrom}]")
        axes.set_xlabel(r"$\lambda$ [\si{\angstrom}]")
        axes.legend(loc="best", frameon=False)
        if title is not None:
            axes.set_title(title)
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    """
    Plot a subset of spectra in the stack

    INPUT:
        indices:    array of indices of spectra in the subset
        fl:         flux(s) to be plotted: ['flux', 'flux_norm', 'flux_interp']
        wl:         wavelength to be plotted: ['loglam', 'loglam_dered']
        linear:     flag to show linear or logarithmic plot
        show:       flag to show or not the plot
        filename:   name of the saved file
    OUTPUT:
        plot of spectra with shared x-axis
    """
    def plotSpectra(self, indices, fl='flux', wl='loglam', linear=True, show=True, filename=None):
        f, axes = plt.subplots(len(indices), 1, sharex=True)
        for (k, index) in enumerate(indices):
            sp = self.spectra[index]
            flux = []
            if isinstance(fl, basestring):
                if fl is 'flux':
                    flux[0] = sp.flux
                elif fl is 'flux_norm':
                    flux[0] = sp.flux_norm
                elif fl is 'flux_interp':
                    flux[0] = sp.flux_interp
            else:
                for t in fl:
                    flux.append(getattr(sp, t))
            if wl is 'loglam':
                lam = sp.loglam
            elif wl is 'loglam_dered':
                lam = sp.loglam_dered
            if linear:
                lam = 10**lam
            for i in range(len(flux)):
                axes[k].plot(lam, flux[i])
            axes[k].yaxis.set_major_locator(MultipleLocator(1.0))
        plt.legend(loc="best")
        if show:
            plt.show()
        if filename is not None:
            plt.savefig(filename)
        return axes


# TODO: rewrite for parallelization
"""
Derive the error on the age and metallicity estimation using the jackknife
method described in Fagioli et al. (2016)

INPUT:
    stack:  Stack instance with N spectra
"""
def error(stack, wr, gs, wl, dw, dc, tp):
    N = stack.N
    spectra = stack.spectra
    age = stack.mean_log_age
    met = stack.mean_metal
    A = np.empty(N)
    Z = np.empty(N)
    for i in range(N):
        spi = spectra[:i] + spectra[i+1:]
        sti = Stack(spi)
        sti.average(wr, gs, wl=wl, dw=dw, dc=dc)
        sti.ppxffit(temp=tp)
        A[i] = sti.mean_log_age
        Z[i] = sti.mean_metal
    k = float(N-1) / N
    sigma_A = np.sqrt(k*np.sum((age-A)**2))
    sigma_Z = np.sqrt(k*np.sum((met-Z)**2))
    return sigma_A, sigma_Z

###############################################################################
# TESTS
###############################################################################

def main1():
    from time import clock
    # load all fits filenames
    spectra_path = './spectra_dustcorr'
    spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]
    wr = [4100, 4700]
    gs = 1.0
    wl = np.array([5577, 6300, 6363])
    dw = 10
    dc = True
    tp = 'kb'
    t = clock()
    stack = Stack(spectra_files)
    print('Elapsed time in stack initialization: %.2f s' % (clock() - t))
    t = clock()
    lam, flux, noise, contrs = stack.average(wr, gs, wl=wl, dw=dw, dc=dc)
    print('Elapsed time in stacking: %.2f s' % (clock() - t))
    t = clock()
    disp = stack.jackknife()
    print('Elapsed time in jackknife: %.2f s' % (clock() - t))
    t = clock()
    corr = stack.correct()
    print('Elapsed time in correct: %.2f s' % (clock() - t))
    t = clock()
    stack.ppxffit(temp=tp)
    print('Elapsed time in ppxffit: %.2f s' % (clock() - t))
    sigma_A, sigma_Z = error(stack, wr, gs, wl, dw, dc, tp)
    print('Elapsed time in error: %.2f s' % (clock() - t))

    print('Mean log age: %f +- %f Gyr' % (stack.mean_log_age, sigma_A))
    print('Mean log age: %f +- %f ' % (stack.mean_metal, sigma_Z))

    # indices = range(5)
    # axes1 = stack.plotSpectra(indices, fl=['noise_norm', 'flux_norm'], wl='loglam_dered', show=False, filename='spectra-normed.png')

    f, axes = plt.subplots(1, 1, figsize=(16.0, 9.0))
    axes.plot(lam, flux, label="stacked flux")
    axes.plot(lam, noise, label="noise")
    axes.plot(lam, disp, label="dispersion")
    axes.plot(lam, corr, label="correction")
    axes.legend(loc="best", frameon=False)
    plt.show()
    # plt.savefig('spectra-stacked.png')

def main2():
    spectra_path = './spectra'
    spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]
    stack = Stack(spectra_files)
    k = len(spectra_files)

    wl = np.array([5577, 6300, 6363])
    dw = 10
    lam, flux, noise, contr = stack.average([4100, 4700], 1.0)
    lam_masked, flux_masked, noise_masked, contr_masked = stack.average([4100, 4700], 1.0, wl=wl, dw=dw)

    f, axes = plt.subplots(2, 1, sharex=True, figsize=(16.0, 9.0))
    ln = [wl-dw,wl+dw]
    axes[0].plot(lam, flux, label="stacked flux")
    axes[0].plot(lam_masked, flux_masked, label="with masking")
    axes[0].plot(lam_masked, noise_masked, label="noise")
    axes[0].vlines(wl, 0, flux.max(), label="sky emission lines")
    # axes[0].vlines(ln, 0, flux.max(), linestyle="dotted")
    axes[0].legend(loc="best", frameon=False, fontsize="12")
    axes[1].vlines(wl, 0, k+2)
    # axes[1].vlines(ln, 0, k+2, linestyle="dotted")
    axes[1].plot(lam, contr_masked, label="contributions")
    axes[1].legend(loc="best", frameon=False, fontsize="12")
    # plt.show()
    plt.savefig('spectra-masking.png')

def main3():
    spectra_path = './spectra_dustcorr'
    spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]
    stack = Stack(spectra_files)
    k = len(spectra_files)

    wl = np.array([5577, 6300, 6363])
    dw = 10
    lam, flux, noise, contr = stack.average([4100, 4700], 1.0, dc=True)
    lam_masked, flux_masked, noise_masked, contr_masked = stack.average([4100, 4700], 1.0, wl=wl, dw=dw, dc=True)

    f, axes = plt.subplots(2, 1, sharex=True, figsize=(16.0, 9.0))
    ln = [wl-dw,wl+dw]
    axes[0].plot(lam, flux, label="stacked flux")
    axes[0].plot(lam_masked, flux_masked, label="with masking")
    axes[0].plot(lam_masked, noise_masked, label="noise")
    axes[0].vlines(wl, 0, flux.max(), label="sky emission lines")
    # axes[0].vlines(ln, 0, flux.max(), linestyle="dotted")
    axes[0].legend(loc="best", frameon=False, fontsize="12")
    axes[1].vlines(wl, 0, k+2)
    # axes[1].vlines(ln, 0, k+2, linestyle="dotted")
    axes[1].plot(lam, contr_masked, label="contributions")
    axes[1].legend(loc="best", frameon=False, fontsize="12")
    # plt.show()
    plt.savefig('spectra-dustcorr-masking.png')

def main4():
    spectra_path = './spectra_dustcorr'
    spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]
    stack = Stack(spectra_files)

    wl = np.array([5577, 6300, 6363])
    dw = 10
    lam, flux, noise, contr = stack.average([4100, 4700], 1.0, wl=wl, dw=dw)
    lam_dc, flux_dc, noise_dc, contr_dc = stack.average([4100, 4700], 1.0, wl=wl, dw=dw, dc=True)

    f, axes = plt.subplots(1, 1, sharex=True, figsize=(16.0, 9.0))
    axes.plot(lam, flux, label="stacked flux")
    axes.plot(lam_dc, flux_dc, label="with dust correction")
    axes.plot(lam, abs(flux_dc-flux), label="difference")
    axes.legend(loc="best", frameon=False, fontsize="12")
    # plt.show()
    plt.savefig('spectra-dustcorr.png')

def main5():
    spectra_path = './spectra_dustcorr'
    spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]

    sp = Spectrum(spectra_files[0])

    f, axes = plt.subplots(1, 1, sharex=True, figsize=(16.0, 9.0))
    axes.plot(10**sp.loglam, sp.flux, label="flux")
    axes.plot(10**sp.loglam, sp.flux_dustcorr, label="with dust correction")
    axes.plot(10**sp.loglam, abs(sp.flux_dustcorr-sp.flux), label="difference")
    axes.legend(loc="best", frameon=False, fontsize="12")
    # plt.show()
    plt.savefig('spectra-comp.png')


if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join
    # main1()
    # main2()
    # main3()
    # main4()
    # main5()
    main6()
