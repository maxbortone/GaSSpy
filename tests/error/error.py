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
spectra_path = path + 'SDSS_spectra/intermediate'
# spectra_path = path + 'SDSS_spectra/old'

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

for (i, f) in enumerate(spectra_files):
    print("Running error test on spectrum #{} from {}".format(i+1, f))
    if i == 5:
        break
    sp = Spectrum(f)
    sp.deredshift()
    sp.normalize(wr, dc)
    sp.signaltonoise(wlr)
    sp.interpolate(gs)
    sp.mask_skylines(wl, dw)

    M = len(sp.lam_interp)
    lam = 10**sp.loglam_dered
    err = sp.error_norm
    error = np.zeros(M)
    for i in range(M):
        x_c = sp.lam_interp[i]
        idx = np.searchsorted(lam, x_c, side='right')
        x_l = lam[idx-1]
        x_r = lam[idx]
        delta = x_r - x_l
        error[i] = np.sqrt((x_r - x_c)**2 * err[idx-1]**2 + (x_c - x_l)**2 * err[idx]**2) / delta

    f, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
    ax.plot(sp.lam_interp, sp.error_interp, label="error interp")
    ax.plot(sp.lam_interp, error, label="error in quad")
    ax.legend(loc="best")
    plt.show()

# f, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
# ax.plot(sp.lam_interp, sp.flux_interp, label="flux")
# ax.plot(sp.lam_interp, sp.error_interp, label="error")
# ax.plot(sp.lam_interp, sp.mask, label="mask")
# ax.legend(loc="best")
# plt.show()
