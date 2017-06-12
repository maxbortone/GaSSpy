import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev
from scipy.integrate import quad
from scipy.special import erf
from astropy.stats import sigma_clip
from spectrum import Spectrum
from ppxf import ppxf
import ppxf_util as util
import miles_util as lib

# gaussian functions for emission line fitting
# a: amplitude
# w: width
# c: centroid
# y: offset
def gaussian(x, a, c, w, y):
    return (a/(np.sqrt(2*np.pi)*w))*np.exp(-(x-c)**2/(2*w**2)) + y

def double_gaussian(x, a1, c1, w1, a2, c2, w2, y):
    return gaussian(x, a1, c1, w1, 0.0) + gaussian(x, a2, c2, w2, 0.0) + y

def triple_gaussian(x, a1, c1, w1, a2, c2, w2, a3, c3, w3, y):
    return gaussian(x, a1, c1, w1, 0.0) + gaussian(x, a2, c2, w2, 0.0) + gaussian(x, a3, c3, w3, 0.0) + y

def Fitzpatrick_EC(wavelength):
    '''
    Fitzpatrick (1999, PASP, 111, 63) extinction curve
    Input:
        wavelength: wavelength in angstrom
    Output:
        k(lambda) = A(lambda)/E(B-V)
    '''
    w_anchor_f99 = np.array([np.inf, 26500., 12200., 6000., 5470.,
                             4670., 4110., 2700., 2600.])  # angstrom
    wnumber_anchor_f99 = 1./(w_anchor_f99 * 1.e-4)  # 1/micron
    alam_ebv_anchor_f99 = np.array([0., 0.265, 0.829, 2.688, 3.055,
                                    3.806, 4.315, 6.265, 6.591])
    tck_f99 = splrep(wnumber_anchor_f99, alam_ebv_anchor_f99, k=3)
    alam_ebv_f99 = splev(1./(wavelength*1e-4), tck_f99, ext=2)
    return alam_ebv_f99


class Stack:
    """
    Initialize stack with a list of spectra

    INPUT:
        filenames:  list of filenames of fits files or Spectrum class instances
        name:       name of the stack instance
    """
    def __init__(self, filenames, name=None):
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
        # initialize attributes
        # name of the stack
        self.name = None
        # length of stacked spectrum
        self.M = None
        # weights by which each spectra is multiplied in the stacking process
        self.weights = None
        # stores the weighted sum of the spectra at each wavelength pixel
        self.P = None
        # stores the sum of the weights of the spectra at each wavelength pixel
        self.S = None
        # wavelength pixels of stacked spectrum
        self.wave = None
        # flux of stacked spectrum
        self.flux = None
        # flux error of stacked spectrum
        self.error = None
        # number of spectra that contributed to each wavelength pixel
        self.contributions = None
        # dispersion of the sample at each wavelength pixel
        self.dispersion = None
        # bias correction
        self.correction = None
        # signal to noise ratio of the stacked spectrum
        self.SNR = None
        # MILES template that is being used for the ppxf fit
        self.template = None
        # ppxf fit result
        self.pp = None
        # difference between stacked spectrum and ppxf bestfit
        self.residual = None
        # gaussian fits to emission lines in the residual
        self.gaussians = {}
        # weights by which each template was multiplied to best fit
        # the stacked spectrum
        self.temp_weights = None
        # mean logarithmic age
        self.mean_log_age = None
        # mean metallicity
        self.mean_metal = None
        # mass to light ratio assuming Salpeter IMF
        # TODO: check if this makes sense with temp='kb'
        self.mass_to_light = None
        # default emission lines to be masked in ppxf
        # (source: http://classic.sdss.org/dr7/algorithms/reflines.dat)
        # NOTE: wavelengths are given in angstrom for light travelling in vacuum
        self.emlines = [('OII', 3728.3), ('H_delta', 4102.89), ('H_gamma', 4341.68), ('H_beta', 4862.68),
                        ('OIII', 4960.295), ('OIII', 5008.24), ('OI', 6302.046),
                        [('NII', 6549.86), ('H_alpha', 6564.61), ('NII', 6585.27)],
                        [('SII', 6718.29), ('SII', 6732.67)]]
        # total fluxes of detected emission lines
        self.flux_emlines = {}

    """
    Prepare the spectra for stacking: deredshift, normalize, compute SNR and
    interpolate flux and error

    INPUT:
        wr:     tuple or array of two floats representing a wavelength range,
                e.g. [4100, 4700], to which the spectra are normalize
        wlr:    list or array of wavelength regions where to compute the SNR
        gs:     spacing of wavelength grid, e.g. 1 angstrom, for the interpolation
        wl:     wavelengths of the sky spectrum to be masked
        dw:     width of region around wl to be masked
        dc:     wheter to use the dust corrected flux or not
    """
    def prepare_spectra(self, wr, wlr, gs, wl=None, dw=None, dc=False):
        # prepare spectra
        for i in range(self.N):
            sp = self.spectra[i]
            sp.deredshift()
            sp.normalize(wr, dc)
            sp.signaltonoise(wlr)
            sp.interpolate(gs)
            if (wl is not None) and (dw is not None):
                sp.mask_skylines(wl, dw)
            sp.mask_sigma_peaks()

    """
    Determine the largest common wavelength range for all spectra
    """
    def determine_wavelength_range(self):
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

    """
    Determine the weights for each spectra in the stack from their SNR
    """
    def determine_weights(self):
        self.weights = np.zeros((self.N, self.M))
        for i in range(self.N):
            sp = self.spectra[i]
            for j in range(self.M):
                if not sp.mask[j]:
                    self.weights[i, j] = sp.flux_interp[j] / sp.error_interp[j]
        for j in range(self.M):
            # normalize weights
            self.weights[:, j] = self.weights[:, j] / sum(self.weights[:, j])

    """
    Calculate the weighted average of the stack

    NOTE:
        run prepare_spectra, determine_wavelength_range and
        determine_weights before average!
    """
    def average(self):
        lam = self.spectra[0].lam_interp
        flux = np.empty(self.M)
        error = np.empty(self.M)
        contrs = np.empty(self.M)
        P = np.empty(self.M)
        S = np.empty(self.M)
        # stack spectra at each wavelength in the common range
        # according to mask of the spectra
        for j in range(self.M):
            p = 0
            s = 0
            n = 0
            c = self.N
            for i in range(self.N):
                sp = self.spectra[i]
                if sp.mask[j]:
                    c -= 1
                else:
                    p += sp.flux_interp[j] * self.weights[i, j]
                    s += self.weights[i, j]
                    n += sp.error_interp[j]**2 * self.weights[i, j]**2
            if s == 0:
                flux[j] = 0
            else:
                P[j] = p
                S[j] = s
                flux[j] = p / s
            error[j] = np.sqrt(n)
            contrs[j] = c
        self.wave = lam
        self.flux = flux
        self.error = error
        self.contributions = contrs
        self.P = P
        self.S = S

    """
    Use the jackknife method to estimate the dispersion in the sample
    """
    def jackknife(self):
        # TODO: assert flux_interp and error_interp are not None
        disp = np.empty(self.M)
        for j in range(self.M):
            p = self.P[j]
            s = self.S[j]
            g = 0
            for i in range(self.N):
                sp = self.spectra[i]
                if not sp.mask[j]:
                    w = self.weights[i, j]
                    f = sp.flux_interp[j]
                    g += ((p*(s-w) - s*(p-w*f)) / (s*(s-w)))**2
            c = self.contributions[j]
            disp[j] = g * (c - 1) / c
        self.dispersion = np.sqrt(disp)
        if self.dispersion.mean() > 0.01*self.flux.mean():
            # TODO: look into
            #   - 6560+-60 (H_alpha emission region)
            #   - continuum at 6800-7200
            # give out percentage difference
            if self.name:
                print("Warning: sample dispersion of {} is high".format(self.name))
            else:
                print("Warning: sample dispersion is high")

    """
    Computes the signal-to-noise ratio of the stacked spectrum

    INPUT:
        wlr: list or array of wavelength regions where to compute the SNR
        flag: boolean, if True outputs SNR value in each wavelength region
    OUTPUT:
        if @flag is set to True it returns the signal-to-noise ratios in the
        wavelength regions defined in @wlr
    """
    def signaltonoise(self, wlr, flag=False):
        m = wlr.shape[0]
        SNRs = np.empty(m)
        S = np.empty(m)
        N = np.empty(m)
        for i in range(m):
            idx = (self.wave >= wlr[i][0]) & (self.wave <= wlr[i][1])
            s = np.median(self.flux[idx])
            n = np.median(self.error[idx])
            SNRs[i] = s / n
            S[i] = s
            N[i] = n
        self.SNR = np.median(S) / np.median(N)
        if flag:
            return SNRs

    """
    Fit the stacked spectrum using the Penalized Pixel Fitting method ppxf by
    Michele Cappellari (http://www-astro.physics.ox.ac.uk/~mxc/software/) with
    the Miles stellar library (http://www.iac.es/proyecto/miles/)

    INPUT:
        temp:       choose between 4 different templates with different IMF,
                    [default, kb, ku, un]
        refit:      refit the stacked spectrum after having removed the residuals
                    in the emission lines
    """
    def ppxffit(self, temp='default', refit=False):
        # CONSTANTS:
        # speed of light in km/s
        c = 299792.458
        # SDSS instrumental resolution
        FWHM_gal = 2.76
        # path to miles stellar library
        path = os.path.join(os.path.dirname(__file__), 'miles_models')
        templates = {
            'default': 'MILES_default/Mun1.30*.fits',
            'kb': 'MILES_Padova00_kb/Mkb1.30*.fits',
            'ku': 'MILES_Padova00_ku/Mku1.30*.fits',
            'un': 'MILES_Padova00_un/Mun1.30*.fits'
        }
        miles_path = os.path.join(path, templates[temp])
        self.template = temp
        # redshift is set to zero since stacked spectrum is calculated
        # in rest frame
        z = 0.0

        if refit:
            if len(self.gaussians) == 0:
                self.fit_residual()
            # galaxy spectrum is already in same wavelength range as stellar library
            wave = np.array(self.pp.lam)
            flux = np.array(self.pp.galaxy)
            for g in self.gaussians.values():
                flux[g['idx']] -= g['gs']
            # only compute velocity scale for ppxf since flux was already log-rebinned
            # (copied from log_rebin function in ppxf_util.py)
            n = len(flux)
            lamRange = np.array([wave[0], wave[-1]])
            dLam = np.diff(lamRange)/(n-1)
            lim = lamRange/dLam + [-0.5, 0.5]
            logLim = np.log(lim)
            velscale = np.diff(logLim)/n*c
            loglam = np.log(wave)
            # select noise
            mask = (self.wave > 3540) & (self.wave < 7409)
            noise = self.error[mask]
        else:
            # run bias correction
            self._correct()
            # bring galaxy spectrum to same wavelength range as stellar library
            mask = (self.wave > 3540) & (self.wave < 7409)
            wave = self.wave[mask]
            flux = self.flux[mask]
            noise = self.error[mask]
            # rebin flux logarithmically for ppxf
            flux, loglam, velscale = util.log_rebin([wave[0], wave[-1]], flux)
            flux = flux / np.median(flux)

        # initialize miles stellar library
        miles = lib.miles(miles_path, velscale, FWHM_gal)
        # determine goodpixels
        lam = np.exp(miles.log_lam_temp)
        dv = c*np.log(lam[0]/wave[0])  # km/s
        lam_range_temp = [lam.min(), lam.max()]
        # compute good pixel by masking emission lines
        # if refit do not mask emission lines
        goodpixels = self._determine_goodpixels(loglam, lam_range_temp, z, refit=refit)
        # initialize start values
        # TODO: check if initial estimate can be optimized to reduce
        # comp time, z should be set to mean redshift of stacked spectrum
        vel = c*np.log(1 + z)   # galaxy velocity in km/s
        start = [vel, 180.]     # starting guess for [V, sigma] in km/s
        # TODO: determine importance and effect of this parameter on
        # the quality of the fit
        delta = 0.004       # regularization error
        # fit stacked spectrum with ppxf
        # NOTE:
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
        temp_weights = pp.weights.reshape(miles.n_ages, miles.n_metal)/pp.weights.sum()
        self.mean_log_age, self.mean_metal = miles.mean_age_metal(temp_weights, quiet=True)
        self.mass_to_light = miles.mass_to_light(temp_weights, band="r", quiet=True)
        self.pp = pp
        self.pp.lam = np.exp(loglam)
        self.temp_weights = temp_weights
        self.residual = self.pp.galaxy - self.pp.bestfit

    """
    Fit residual spectrum at emission lines with gaussians
    """
    def fit_residual(self):
        wv = self.pp.lam
        rs = self.residual
        for el in self.emlines:
            if isinstance(el, tuple):
                line = el[1]
                name = el[0]
                self._fit_singlet(wv, rs, line, name)
            elif isinstance(el, list) and len(el) == 2:
                doublet = [el[0][1], el[1][1]]
                name = el[0][0] + '-' + el[1][0]
                self._fit_doublet(wv, rs, doublet, name)
            elif isinstance(el, list) and len(el) == 3:
                triplet = [el[0][1], el[1][1], el[2][1]]
                name = el[0][0] + '-' + el[1][0] + '-' + el[2][0]
                self._fit_triplet(wv, rs, triplet, name)

    """
    Set the total flux of the detected emission lines

    NOTE:
        Shifting the gaussian fits vertically by -y (offset) and integrating from
        -inf to +inf yields a (amplitude) as the total flux
    """
    def set_emlines_flux(self):
        for (key, val) in self.gaussians.items():
            parts = key.split('-')
            for (i, p) in enumerate(parts):
                ia = i*3
                f = val['popt'][ia]
                e = np.sqrt(val['pcov'][ia][ia])
                self.flux_emlines[p] = (f, e)

    """
    Estimate dust attenuation from H_alpha and H_beta total flux and correct
    stacked spectrum using the Fitzpatrick extinction curve

    NOTE:
        assume Calzetti et al. 2000 extinction curve with R_V = 3.1 to estimate
        gas color excess. Compute the star color excess with the fudge factor 0.44
    """
    def correct_dust_attenuation(self):
        # compute color excess of gas
        # - H_alpha can be detected as singlet, doublet with NII or as triplet
        #   with NII on the left and on the right
        # - H_beta is detected as singlet
        F_Ha = None
        F_Hb = None
        for (key, val) in self.flux_emlines.items():
            if 'H_alpha' in key:
                F_Ha = val[0]
            if 'H_beta' in key:
                F_Hb = val[0]
        if F_Ha == None and F_Hb == None:
            print("No emission lines detected.")
            return
        elif F_Ha == None:
            print("No H_alpha line detected.")
            return
        elif F_Hb == None:
            print("No H_beta line detected.")
            return
        E_gas = 1.97*np.log10(F_Ha/F_Hb/2.86)
        # compute color excess of stars
        E_stars = 0.44*E_gas
        print("\t dust attenuation: {}".format(E_stars))
        # correct flux with Fitzpatrick extinction curve
        correction = 0.4*E_stars*Fitzpatrick_EC(self.wave)
        self.flux *= 10**(correction)
        idx = (self.wave >= 4100) & (self.wave <= 4700)
        self.flux /= np.median(self.flux[idx])

    """
    Plot stacked spectrum, noise, dispersion and bias correction

    INPUT:
        filename:   plot filename
        title:      plot title
    OUTPUT:
        plot of stacked spectra
    """
    def plotStack(self, filename=None, title=None):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        # setup matplotlib
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = [
               r'\usepackage{siunitx}',
               r'\DeclareSIUnit\ergs{ergs}',
               r'\sisetup{per-mode=symbol}'
        ]
        wv = self.wave
        f, axes = plt.subplots(3, 1, sharex=True, figsize=(16, 9))
        if self.correction is not None:
            axes[0].text(0.75, 0.85, "bias correction: ON", bbox=dict(facecolor='red', alpha=0.5), transform=axes[0].transAxes)
        else:
            axes[0].text(0.75, 0.85, "bias correction: OFF", bbox=dict(facecolor='red', alpha=0.5), transform=axes[0].transAxes)
        axes[0].plot(wv, self.flux, 'b', label="stacked flux")
        axes[0].plot(wv, self.error, 'g', label="error")
        axes[0].set_ylabel(r"flux [\SI{e-17}{\ergs\per\second\per\square\centi\meter\per\angstrom}]")
        axes[0].legend(loc="upper left", frameon=False)
        axes[1].plot(wv, self.dispersion, 'r', label="dispersion")
        axes[1].plot(wv, self.correction, 'y', label="bias correction")
        axes[1].set_ylabel(r"flux [\SI{e-17}{\ergs\per\second\per\square\centi\meter\per\angstrom}]")
        axes[1].legend(loc="upper left", frameon=False)
        axes[2].plot(wv, self.contributions, label="contributions")
        axes[2].set_ylabel("number of spectra")
        axes[2].set_xlabel(r"wavelength [\si{\angstrom}]")
        axes[2].set_ylim([self.contributions.min()-5, self.N+5])
        axes[2].yaxis.set_major_locator(MultipleLocator(5))
        axes[2].legend(loc="lower left", frameon=False)
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
    OUTPUT:
        plot of bestfit spectra obtained from ppxf
    """
    def plotFit(self, filename=None, title=None):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        # setup matplotlib
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = [
               r'\usepackage{siunitx}',
               r'\DeclareSIUnit\ergs{ergs}',
               r'\sisetup{per-mode=symbol}'
        ]
        wv = self.pp.lam
        f, axes = plt.subplots(1, 1, figsize=(11.69,8.27))
        axes.plot(wv, self.pp.galaxy, 'b', label="stacked spectrum")
        axes.plot(wv, self.pp.bestfit, 'g', label="ppxf fit")
        axes.plot(wv, self.residual, 'r', label="residual")
        if self.SNR:
            axes.text(0.75, 0.05, "SNR = {:.2f}".format(self.SNR), bbox=dict(facecolor='red', alpha=0.5), transform=axes.transAxes)
        axes.set_ylabel(r"flux [\SI{e-17}{\ergs\per\second\per\square\centi\meter\per\angstrom}]")
        axes.set_xlabel(r"wavelength [\si{\angstrom}]")
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
        fl:         flux to be plotted: ['flux', 'flux_norm', 'flux_interp']
        wl:         wavelength to be plotted: ['loglam', 'loglam_dered']
        show:       flag to show or not the plot
        filename:   name of the saved file
    OUTPUT:
        plot of spectra with shared x-axis
    """
    def plotSpectra(self, indices, fl='flux', wl='loglam', wr=None, show=True, title=None, filename=None):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        # setup matplotlib
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = [
               r'\usepackage{siunitx}',
               r'\DeclareSIUnit\ergs{ergs}',
               r'\sisetup{per-mode=symbol}'
        ]
        f, axes = plt.subplots(len(indices), 1, figsize=(11.69,8.27), sharex=True)
        for (k, index) in enumerate(indices):
            sp = self.spectra[index]
            if fl is 'flux':
                flux = sp.flux
                error = sp.error
            elif fl is 'flux_norm':
                flux = sp.flux_norm
                error = sp.error_norm
            elif fl is 'flux_interp':
                flux = sp.flux_interp
                error = sp.error_interp
            if wl is 'loglam':
                lam = 10**sp.loglam
            elif wl is 'loglam_dered':
                lam = 10**sp.loglam_dered
            elif wl is 'lam_interp':
                lam = sp.lam_interp
            if wr:
                idx = (lam >= wr[0]) & (lam <= wr[1])
                axes[k].plot(lam[idx], flux[idx])
                axes[k].plot(lam[idx], error[idx])
            else:
                axes[k].plot(lam, flux)
                axes[k].plot(lam, error)
            axes[k].yaxis.set_major_locator(MultipleLocator(1.0))
            axes[k].set_ylim(0.0, 2.0)
            axes[k].text(0.05, 0.85, "Spectrum: {}".format(index+1), verticalalignment="top", transform=axes[k].transAxes)
        f.text(0.5, 0.04, 'wavelength', ha='center', va='center')
        f.text(0.06, 0.5, 'flux', ha='center', va='center', rotation='vertical')
        f.suptitle(title)
        # plt.subplots_adjust(bottom=0.01)
        # plt.legend(loc="best")
        if show:
            plt.show()
        if filename is not None:
            plt.savefig(filename)
        return axes

    ###########################################################################
    # PRIVATE METHODS
    ###########################################################################
    """
    Correct the stacked spectrum for possible sampling bias
    """
    def _correct(self):
        corr = np.zeros(self.M)
        for j in range(self.M):
            p = self.P[j]
            s = self.S[j]
            c = self.contributions[j]
            for i in range(self.N):
                sp = self.spectra[i]
                if not sp.mask[j]:
                    w = self.weights[i, j]
                    f = sp.flux_interp[j]
                    if s != w:
                        corr[j] += (p - w*f) / (s - w)
            corr[j] = (c-1)*(corr[j]/c - self.flux[j])
        self.correction = corr
        self.flux -= self.correction

    """
    # forked from ppxf_util.py by Michele Cappellari
    Generates a list of goodpixels to mask a given set of gas emission
    lines. This is meant to be used as input for PPXF.

    INPUT:
        logLam:         natural logarithm np.log(wave) of the wavelength in
                        Angstrom of each pixel of the log rebinned galaxy spectrum
        lamRangeTemp:   two elements vectors [lamMin, lamMax] with the minimum
                        and maximum wavelength in Angstrom in the stellar template used in PPXF
        z:              estimate of the galaxy redshift
        refit:          bool, wheter the ppxf fit is being run a second time or not
    OUTPUT:
        goodpixels:     vector of goodPixels to be used as input for PPXF
    """
    def _determine_goodpixels(self, logLam, lamRangeTemp, z, refit=False):
        # speed of light in km/s
        c = 299792.458
        flag = np.zeros_like(logLam, dtype=bool)
        if not refit:
            lines = []
            for el in self.emlines:
                if isinstance(el, tuple):
                    lines.append(el[1])
                elif isinstance(el, list):
                    for l in el:
                        lines.append(l[1])
            # width/2 of masked gas emission region in km/s
            dv = np.ones(len(lines))*800
            for line, dvj in zip(lines, dv):
                flag |= (np.exp(logLam) > line*(1 + z)*(1 - dvj/c)) \
                      & (np.exp(logLam) < line*(1 + z)*(1 + dvj/c))

        # mask edges of stellar library
        flag |= np.exp(logLam) > lamRangeTemp[1]*(1 + z)*(1 - 900/c)
        flag |= np.exp(logLam) < lamRangeTemp[0]*(1 + z)*(1 + 900/c)

        return np.where(flag == 0)[0]

    """
    Detect the presence of a peak at line in @lines using sigma clipping. A peak is
    detected if a positive 1sigma outlier is found within 3 angstrom from line

    INPUT:
        x:     array containing the x coordinate of the signal
        y:     array containing the y coordinate of the signal
        lines: float or list, contains the expected positions of the peak(s)
               (in ascending order)

    OUTPUT:
        detected:   boolean or array of booleans, True if peak was detected
    """
    def _detect_peak(self, x, y, lines):
        if isinstance(lines, float):
            lines = [lines]
        detected = np.zeros_like(lines, dtype=bool)
        # select 20 pixel intervall around emission line
        idx = (x >= lines[0]-20) & (x <= lines[-1]+20)
        x = x[idx]
        y = y[idx]
        sc = sigma_clip(y, sigma=1, iters=None)
        py = sc.data[sc.mask]
        px = x[sc.mask]
        for i in range(len(lines)):
            # check if there is a peak within 3 pixels from emission line
            # and if it is positive
            pidx = (px >= lines[i] - 3) & (px <= lines[i] + 3)
            if pidx.any() and (py[pidx] > 0).all():
                detected[i] = True
        if len(detected) == 1:
            return detected[0]
        else:
            return detected

    """
    Fit a signal around @line with a simple gaussian curve

    INPUT:
        x:     array containing the x coordinate of the signal
        y:     array containing the y coordinate of the signal
        line:  float, wavelength around which fit should be performed
    """
    def _fit_singlet(self, x, y, line, name):
        gx = None
        gs = None
        idx = None
        if line >= x[0] and line <= x[-1]:
            det = self._detect_peak(x, y, line)
            print("\tpeak at: {} = {}".format(line, det))
            if det:
                idx = (x >= line - 10) & (x <= line + 10)
                gx = x[idx]
                gy = y[idx]
                params = [1.0, 0.0, 2.0, 0.0]
                try:
                    popt, pcov = curve_fit(gaussian, gx-line, gy, p0=params)
                except RuntimeError as err:
                    print("Curve fitting the line at {} raised runtime error: {}".format(line, err))
                    return
                else:
                    gs = gaussian(gx-line, *popt)
                    self.gaussians[name] = {'type': 'singlet', 'line': line,
                                            'popt': popt, 'pcov': pcov,
                                            'idx': idx, 'gx': gx, 'gs': gs}

    """
    Fit a signal around between the lines in @doublet with a double gaussian curve

    INPUT:
        x:        array containing the x coordinate of the signal
        y:        array containing the y coordinate of the signal
        doublet:  list containing the wavelenghts of the lines around which
                  the fit should be performed (in ascending order)
    """
    def _fit_doublet(self, x, y, doublet, name):
        gx = None
        gs = None
        idx = None
        if doublet[0] >= x[0] and doublet[1] <= x[-1]:
            det = self._detect_peak(x, y, doublet)
            print("\tpeak at: {} = {}, {} = {}".format(doublet[0], det[0], doublet[1], det[1]))
            if det[0] and det[1]:
                idx = (x >= doublet[0] - 10) & (x <= doublet[1] + 10)
                gx = x[idx]
                gy = y[idx]
                c0 = (doublet[1]-doublet[0])/2
                params = [1.0, -c0, 2.0, 1.0, c0, 2.0, 0.0]
                try:
                    popt, pcov = curve_fit(double_gaussian, gx-np.mean(doublet), gy, p0=params)
                except RuntimeError as err:
                    print("Curve fitting the doublet at {}-{} raised runtime error: {}"
                          .format(doublet[0], doublet[1], err))
                    return
                else:
                    gs = double_gaussian(gx-np.mean(doublet), *popt)
                    self.gaussians[name] = {'type': 'doublet', 'line': np.mean(doublet),
                                            'popt': popt, 'pcov': pcov,
                                            'idx': idx, 'gx': gx, 'gs': gs}
            if det[0] and not det[1]:
                name = name.split('-')[0]
                self._fit_singlet(x, y, doublet[0], name)
            if det[1] and not det[0]:
                name = name.split('-')[1]
                self._fit_singlet(x, y, doublet[1], name)

    """
    Fit a signal around @line with a simple gaussian curve

    INPUT:
        x:        array containing the x coordinate of the signal
        y:        array containing the y coordinate of the signal
        triplet:  list containing the wavelenghts of the lines around which
                  the fit should be performed (in ascending order)
    """
    def _fit_triplet(self, x, y, triplet, name):
        gx = None
        gs = None
        idx = None
        if triplet[0] >= x[0] and triplet[2] <= x[-1]:
            det = self._detect_peak(x, y, triplet)
            print("\tpeak at: {} = {}, {} = {}, {} = {}".format(triplet[0], det[0], triplet[1], det[1], triplet[2], det[2]))
            if det.all():
                idx = (x >= triplet[0] - 10) & (x <= triplet[2] + 10)
                gx = x[idx]
                gy = y[idx]
                c0 = triplet[1]-triplet[0]
                params = [1.0, -c0, 2.0, 1.0, 0.0, 2.0, 1.0, c0, 2.0, 0.0]
                try:
                    popt, pcov = curve_fit(triple_gaussian, gx-triplet[1], gy, p0=params)
                except RuntimeError as err:
                    print("Curve fitting the triplet at {}-{}-{} raised runtime error: {}"
                          .format(triplet[0], triplet[1], triplet[2], err))
                    return
                else:
                    gs = triple_gaussian(gx-triplet[1], *popt)
                    self.gaussians[name] = {'type': 'triplet', 'line': triplet[1],
                                            'popt': popt, 'pcov': pcov,
                                            'idx': idx, 'gx': gx, 'gs': gs}
            if (det[0] and det[1]) and not det[2]:
                name = name.split('-')[0] + '-' + name.split('-')[1]
                self._fit_doublet(x, y, [triplet[0], triplet[1]], name)
            if (det[1] and det[2]) and not det[0]:
                name = name.split('-')[1] + '-' + name.split('-')[2]
                self._fit_doublet(x, y, [triplet[1], triplet[2]], name)
            if det[0] and (not det[1] and not det[2]):
                name = name.split('-')[0]
                self._fit_singlet(x, y, triplet[0], name)
            if det[1] and (not det[2] and not det[0]):
                name = name.split('-')[1]
                self._fit_singlet(x, y, triplet[1], name)
            if det[2] and (not det[0] and not det[1]):
                name = name.split('-')[2]
                self._fit_singlet(x, y, triplet[2], name)

    """
    Compute the total flux of the detected emission lines by integrating the
    gaussian fitting curves. Also estimate the error on the integral by propagating
    the errors on the fitting parameters through the analytic solution of the integral

    NOTE:
        scipy.integrate.quad gives same result as analytic solutions for the integrals
        of gaussian, double_gaussian and triple_gaussian functions
    """
    def _integrate_gaussians(self):
        for g in self.gaussians.values():
            if g['type'] == 'singlet':
                integrand = lambda x, p: gaussian(x, *p)
            elif g['type'] == 'doublet':
                integrand = lambda x, p: double_gaussian(x, *p)
            elif g['type'] == 'triplet':
                integrand = lambda x, p: triple_gaussian(x, *p)
            a = g['gx'][0] - g['line']
            b = g['gx'][-1] - g['line']
            # parameters are defined as:
            # - singlet: [a, c, w, y]
            # - doublet: [a1, c1, w1, a2, c2, w2, y]
            # - triplet: [a1, c1, w1, a2, c2, w2, a3, c3, w3, y]
            # where a is amplitude, c is centroid, w is width and y is offset
            p = g['popt']
            g['integral'] = quad(integrand, a, b, args=(p,))[0]
            # compute parameter errors from covariance matrix of gaussian fit
            e = np.sqrt(np.diag(g['pcov']))
            e_acw = 0
            for i in range(len(e)/3):
                ia = i*3
                ic = ia+1
                iw = ia+2
                e_a = (erf((b-p[ic])/(np.sqrt(2)*p[iw])) - erf((a-p[ic])/(np.sqrt(2)*p[iw])))/2
                e_c = p[ia]*(-np.sqrt(2/(np.pi*p[iw]))*np.exp(-((b-p[ic])/(np.sqrt(2)*p[iw]))**2)
                            +np.sqrt(2/(np.pi*p[iw]))*np.exp(-((a-p[ic])/(np.sqrt(2)*p[iw]))**2))/2
                e_w = p[ia]*(-np.sqrt(2/np.pi)*((b-p[ic])/(p[iw]**2))*np.exp(-((b-p[ic])/(np.sqrt(2)*p[iw]))**2)
                            +np.sqrt(2/np.pi)*((a-p[ic])/(p[iw]**2))*np.exp(-((a-p[ic])/(np.sqrt(2)*p[iw]))**2))/2
                e_acw += (e_a*e[ia])**2 + (e_c*e[ic])**2 + (e_w*e[iw])**2
            g['integral_error'] = np.sqrt(e_acw + ((b-a)*e[-1])**2)

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


if __name__ == "__main__":
    from time import clock
    path = os.path.dirname(__file__).split('lib')[0]
    # import fits file and initialize stack
    spectra_path = path + 'spectra_dustcorr/'

    # initialize stack
    spectra_files = [os.path.join(spectra_path, f) for f in os.listdir(spectra_path) if os.path.isfile(os.path.join(spectra_path, f))]
    print("Stacking {} spectra".format(len(spectra_files)))

    # compute stacked spectrum with masking and dust correction
    wr = [4100, 4700]                   # wavelength range for normalization
    wlr = np.array([[4500, 4700],       # wavelength regions for SNR analysis
                    [5400, 5600],
                    [6000, 6200],
                    [6800, 7000]])
    gs = 1.0                            # grid spacing for interpolation
    wl = np.array([5577, 6300, 6363])   # sky lines for masking
    dw = 3                              # broadness of sky lines for masking
    dc = True                           # flag for dust correction
    tp = 'kb'                           # Kroupa IMF

    # initialize stack instance
    t = clock()
    stack = Stack(spectra_files)
    print("- stack initialization: {}s".format(clock()-t))
    # prepare spectra for stacking
    t = clock()
    stack.prepare_spectra(wr, wlr, gs, wl=wl, dw=dw, dc=dc)
    print("- spectra preparation: {}s".format(clock()-t))
    # determine common wavelength range
    t = clock()
    stack.determine_wavelength_range()
    print("- wavelength range: {}".format(clock()-t))
    # determine weights from signal-to-noise ratio at each pixel
    t = clock()
    stack.determine_weights()
    print("- determine weights: {}s".format(clock()-t))
    # stack
    t = clock()
    stack.average()
    print("- stacking: {}s".format(clock()-t))
    # compute dispersion of the spectra in the stack using the jackknife method
    t = clock()
    stack.jackknife()
    print("- dispersion: {}s".format(clock()-t))
    # fit stacked spectrum with ppxf
    t = clock()
    stack.ppxffit(temp=tp)
    print("- ppxf fit: {}s".format(clock()-t))
    # compute SNR of stacked spectrum
    stack.signaltonoise(wlr)
    print("- stack SNR = {}".format(stack.SNR))
    # plot results
    stack.plotStack()
    stack.plotFit()
