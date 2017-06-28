from time import clock
import sys, os
path = os.path.dirname(__file__).split('test')[0]
sys.path.append(path + "lib/")
import numpy as np
from spectrum import Spectrum
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('fivezerosix')

# import fits file and initialize stack
# spectra_path = path + 'spectra_dustcorr/'
# spectra_path = path + 'SDSS_spectra/young'
# spectra_path = path + 'SDSS_spectra/intermediate'
# spectra_path = path + 'SDSS_spectra/old'
spectra_path = path + 'SDSS_spectra/bin_15'

spectra_files = [os.path.join(spectra_path, f) for f in os.listdir(spectra_path) if os.path.isfile(os.path.join(spectra_path, f))]

# calculate stacked spectrum with masking and dust correction
wr = [4100, 4700]                   # wavelength range for normalization
wlr = np.array([[4500, 4700],       # wavelength regions for SN analysis
                [5400, 5600],
                [6000, 6200],
                [6800, 7000]])
gs = 1.0                            # grid spacing for interpolation
wl = np.array([5577, 6300, 6363])   # sky lines for masking
dw = 3                              # broadness of sky lines for masking
dc = True                           # flag for dust correction
tp = 'kb'                           # Kroupa IMF

sp = Spectrum(spectra_files[0])
sp.normalize(wr, dc)
wave = 10**sp.loglam

f, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
ax.plot(wave, sp.flux, label="flux")
ax.plot(wave, sp.error, label="error")
ax.plot(wave, sp.ivar, label="inverse variance")
ax.set_ylabel(r"flux [\SI{e-17}{\ergs\per\second\per\square\centi\meter\per\angstrom}]")
ax.set_xlabel(r"wavelength [\si{\angstrom}]")
ax.legend(loc="best")
plt.show()
