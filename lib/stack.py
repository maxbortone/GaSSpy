import os
import numpy as np
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from spectrum import Spectrum
from ppxf import ppxf
import ppxf_util as util
import miles_util as lib
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

def gaussian(x, amp, cen, wid):
    return (amp/(np.sqrt(2*np.pi)*wid))*np.exp(-(x-cen)**2/(2*wid**2))


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
        # initialize attributes
        # name of the stack
        self.name = None
        # wheter sky lines are being masked or not
        self.masking = None
        # length of stacked spectrum
        self.M = None
        # weights by which each spectra is multiplied in the stacking process
        self.weights = np.empty(self.N)
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
        # flux error of stacked spectrum
        self.error = None
        # number of spectra that contributed to each wavelength pixel
        self.contributions = None
        # dispersion of the sample at each wavelength pixel
        self.dispersion = None
        # signal to noise ratio of the stacked spectrum
        self.SNR = None
        # MILES template that is being used for the ppxf fit
        self.template = None
        # ppxf fit result
        self.pp = None
        # difference between stacked spectrum and ppxf bestfit
        self.residual = None
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
        # set masking flag
        self.masking = (wl is not None) and (dw is not None)
        # prepare spectra
        for i in range(self.N):
            sp = self.spectra[i]
            sp.deredshift()
            sp.normalize(wr, dc)
            sp.signaltonoise(wlr)
            sp.interpolate(gs)
            if self.masking:
                sp.setmask(wl, dw)

    """
    Determine the weights for eah spectra in the stack from their SNR
    """
    def determine_weights(self):
        for i in range(self.N):
            sp = self.spectra[i]
            self.weights[i] = sp.SNR
        # normalize weights
        self.weights = self.weights / sum(self.weights)

    """
    Calculate the weighted average of the stack

    NOTE: run prepare_spectra and determine_weights before average!

    OUTPUT:
        lam:    wavelenghts of stacked spectrum
        flux:   flux of stacked spectrum
        error:  flux error of stacked spectrum
        contrs: contributions from spectra to flux at each wavelength
    """
    def average(self):
        # # set masking flag
        # self.masking = (wl is not None) and (dw is not None)
        # # prepare spectra
        # for i in range(self.N):
        #     sp = self.spectra[i]
        #     sp.deredshift()
        #     sp.normalize(wr, dc)
        #     sp.interpolate(gs)
        #     if self.masking:
        #         sp.setmask(wl, dw)
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
        error = np.empty(self.M)
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
                    p += sp.flux_interp[i] * self.weights[j]
                    s += self.weights[j]
                    n += sp.error_interp[i]**2 * self.weights[j]**2
            if s == 0:
                flux[i] = 0
            else:
                P[i] = p
                S[i] = s
                flux[i] = p / s
            error[i] = np.sqrt(n)
            contrs[i] = c
        self.wave = lam
        self.flux = flux
        self.error = error
        self.contributions = contrs
        self.P = P
        self.S = S
        return lam, flux, error, contrs

    """
    Use the jackknife method to estimate the dispersion in the sample
    """
    def jackknife(self):
        # TODO: assert flux_interp and error_interp are not None
        disp = np.empty(self.M)
        for i in range(self.M):
            p = self.P[i]
            s = self.S[i]
            g = 0
            for j in range(self.N):
                sp = self.spectra[j]
                if not (self.masking and sp.mask[i]):
                    w = self.weights[j]
                    f = sp.flux_interp[i]
                    g += ((p*(s-w) - s*(p-w*f)) / (s*(s-w)))**2
            c = self.contributions[i]
            disp[i] = g * (c - 1) / c
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
        temp:   choose between 4 different templates with different IMF,
                [default, kb, ku, un]
        refit:  refit the stacked spectrum after having removed the residuals
                in the Balmer lines H_delta, H_gamma, H_beta
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

        if refit:
            # TODO: assert ppxf fit has already been done once
            # galaxy spectrum is already in same wavelength range as stellar library
            wave = np.array(self.pp.lam)
            flux = np.array(self.pp.galaxy)
            residual = np.array(self.residual)
            #               H_delta, H_gamma, H_beta
            balmer_lines = [4101.76, 4340.47, 4861.33]
            flux, gaussians = self._subtract_residual(wave, flux, residual, balmer_lines)
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
        # if refit do not mask balmer lines
        # TODO: refactor util.determine_goodpixels in stacking.py
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
        temp_weights = pp.weights.reshape(miles.n_ages, miles.n_metal)/pp.weights.sum()
        self.mean_log_age, self.mean_metal = miles.mean_age_metal(temp_weights, quiet=True)
        self.mass_to_light = miles.mass_to_light(temp_weights, band="r", quiet=True)
        self.pp = pp
        self.pp.lam = np.exp(loglam)
        self.temp_weights = temp_weights
        self.residual = self.pp.galaxy - self.pp.bestfit
        if refit:
            self.gaussians = gaussians

    """
    Plot stacked spectrum, noise, dispersion and bias correction

    INPUT:
        filename:   plot filename
        title:      plot title
    """
    def plotStack(self, filename=None, title=None):
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
    """
    def plotFit(self, filename=None, title=None):
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
        f, axes = plt.subplots(len(indices), 1, figsize=(11.69,8.27), sharex=True)
        for (k, index) in enumerate(indices):
            sp = self.spectra[index]
            if isinstance(fl, basestring):
                if fl is 'flux':
                    flux = sp.flux
                elif fl is 'flux_norm':
                    flux = sp.flux_norm
                elif fl is 'flux_interp':
                    flux = sp.flux_interp
            if wl is 'loglam':
                lam = sp.loglam
            elif wl is 'loglam_dered':
                lam = sp.loglam_dered
            lam = 10**lam
            if wr:
                idx = (lam >= wr[0]) & (lam <= wr[1])
                axes[k].plot(lam[idx], flux[idx])
            else:
                axes[k].plot(lam, flux)
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
        for i in range(self.M):
            p = self.P[i]
            s = self.S[i]
            c = self.contributions[i]
            for j in range(self.N):
                sp = self.spectra[j]
                if not (self.masking and sp.mask[i]):
                    w = self.weights[j]
                    f = sp.flux_interp[i]
                    if s != w:
                        corr[i] += (p - w*f) / (s - w)
            corr[i] = (c-1)*(corr[i]/c - self.flux[i])
        self.correction = corr
        self.flux -= self.correction

    """
    Detect the presence of a peak at @line using sigma clipping. A peak is
    detected if a positive 3sigma outlier is found within 3 angstrom from @line

    INPUT:
        wv:     array containing the wavelength coordinate of the spectrum
        fl:     array containing the flux coordinate of the spectrum
        line:   float representing the position of the peak

    OUTPUT:
        detected:   boolean, True if peak was detected
    """
    def _detect_peak(self, wv, fl, line):
        detected = False
        # select 20 pixel intervall around balmer line
        idx = (wv >= line-20) & (wv <= line+20)
        wv = wv[idx]
        fl = fl[idx]
        # perform sigma clipping with 3sigma for upper and lower clipping
        # TODO: check how much should sigma be
        sc = sigma_clip(fl, sigma=3, iters=None)
        pfl = sc.data[sc.mask]
        pwv = wv[sc.mask]
        # check if there is a peak within 3 pixels from balmer line
        # and if it is positive
        pidx = (pwv >= line - 3) & (pwv <= line + 3)
        if pidx.any() & (pfl[pidx] > 0).all():
            detected = True
        return detected

    """
    Subtract residual from stacked spectrum at Balmer lines by fitting detected
    peaks with gaussians

    INPUT:
        wv:     array containing the wavelength coordinate of the spectrum
        fl:     array containing the flux coordinate of the spectrum
        rs:     array containing the residual to the bestfit of the spectrum
        lines:  list of emission lines, e.g. Balmer lines

    OUTPUT:
        fl_corr:    array containing the residual subtracted flux
        gaussians:  list of fitted gaussians
    """
    def _subtract_residual(self, wv, fl, rs, lines):
        fl_corr = np.array(fl)
        gaussians = []
        for i, line in enumerate(lines):
            detected = self._detect_peak(wv, rs, line)
            if detected:
                idx = (wv >= line - 10) & (wv <= line + 10)
                gwv = wv[idx] - line
                grs = rs[idx]
                popt, pcov = curve_fit(gaussian, gwv, grs, p0=[1.0, 0.0, 2.0])
                gs = gaussian(gwv, popt[0], popt[1], popt[2])
                fl_corr[idx] -= gs
                gaussians.append(np.array([gwv+line, gs]))
        return fl_corr, gaussians

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
    t = clock()
    stack = Stack(spectra_files)
    print("- stack initialization: {}s".format(clock()-t))

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

    # prepare spectra for stacking
    t = clock()
    stack.prepare_spectra(wr, wlr, gs, wl=wl, dw=dw, dc=dc)
    print("- prepare spectra: {}s".format(clock()-t))
    # determine weights from SNR
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

    # plot results
    stack.plotStack()
    stack.plotFit()
