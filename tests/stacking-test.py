from os import listdir
from os.path import isfile, join
from math import floor
from numpy import log10, arange, interp
from numpy.ma import masked_outside
from astroquery.sdss import SDSS
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['mathtext.fontset'] = 'stixsans'
mpl.rcParams['font.family'] = 'sans'
mpl.rcParams['font.serif'] = 'STIXGeneral'
mpl.rcParams['text.usetex'] =True
mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',     # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',     # ...this to force siunitx to actually use your fonts
       r'\usepackage{sans}',        # set the normal font here
       r'\usepackage{sansmath}',    # load up the sansmath so that math -> helvet
       r'\sansmath',                # <- tricky! -- gotta actually tell tex to use!
       r'\DeclareSIUnit\erg{erg}'
]

def plot_spectra(spec):
    tbdata = spec[1].data
    flux = tbdata['flux']
    loglam = tbdata['loglam']
    loglam_dered = deredshift(spec)
    f, axes = plt.subplots(1, 1, figsize=(7.0, 5.0))
    axes.plot(10**loglam, flux, label="coadded calibrated flux")
    axes.plot(10**loglam_dered, flux, label="deredshifted coadded calibrated flux")
    axes.set_ylabel(r"$F_{\lambda}$ [\SI{e-17}{\erg\per\second\per\square\centi\meter\per\angstrom}]")
    axes.set_xlabel(r"$\lambda$ [\si{\angstrom}]")
    axes.legend(loc="best", frameon=False)

def deredshift_spectra(spec, redshift=False):
    spec_data = spec[1].data
    spec_meta = spec[2].data
    loglam = spec_data['loglam']
    z = spec_meta['Z']
    loglam_dered = loglam - log10(1+z)
    if redshift:
        return loglam_dered, z
    else:
        return loglam_dered

def normalize_spectra(spec):
    spec_data = spec[1].data
    flux = spec_data['flux']
    loglam = spec_data['loglam']
    ma = masked_outside(10**loglam, 4100, 4700)
    ind = ~ma.mask
    flux_selection = flux[ind]
    mean_flux = flux_selection.mean()
    return flux / mean_flux

def interpolate_spectra(spec):
    flux = normalize_spectra(spec)
    lam = 10**deredshift_spectra(spec)
    a = floor(lam[0])+1
    b = floor(lam[-1])+1
    lam_interp = arange(a, b, 1.0)
    flux_interp = interp(lam_interp, lam, flux)
    return flux_interp, lam_interp

# load all fits filenames
spectra_path = './spectra'
spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]

f, axes = plt.subplots(5, 1, sharex=True)
# axes[0].set_xlabel(r"$\lambda$ [\si{\angstrom}]")
for j in range(len(spectra_files[:5])):
    spec = fits.open(spectra_files[j])
    spec_data = spec[1].data
    flux = spec_data['flux']
    loglam = spec_data['loglam']
    loglam_dered, z = deredshift_spectra(spec, redshift=True)
    flux_norm = normalize_spectra(spec)
    flux_interp, lam_interp = interpolate_spectra(spec)
    # axes[j].plot(10**loglam, flux, label="coadded calibrated flux")
    # axes[j].plot(10**loglam_dered, flux, label="deredshifted coadded calibrated flux")
    axes[j].plot(10**loglam_dered, flux_norm, label="deredshifted normalized coadded calibrated flux")
    axes[j].plot(lam_interp, flux_interp, label="interpolated flux")
    # redshift_str = r"z = %.4f"%(z)
    # axes[j].text(0.95, 0.5, redshift_str, verticalalignment='bottom', horizontalalignment='right', transform=axes[j].transAxes)
    # axes[j].set_ylabel(r"$F_{\lambda}$ [\SI{e-17}{\erg\per\second\per\square\centi\meter\per\angstrom}]")
    # axes[j].legend(loc="best", frameon=False)

plt.show()
