from time import clock
import sys, os
path = os.path.dirname(__file__).split('test')[0]
sys.path.append(path + "lib/")
import numpy as np
from stack import Stack
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',
       r'\DeclareSIUnit\ergs{ergs}',
       r'\sisetup{per-mode=symbol}'
]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# import fits file and initialize stack
# spectra_path = path + 'spectra_dustcorr/'
# spectra_path = path + 'SDSS_spectra/young'
spectra_path = path + 'SDSS_spectra/intermediate'
# spectra_path = path + 'SDSS_spectra/old'

spectra_files = [os.path.join(spectra_path, f) for f in os.listdir(spectra_path) if os.path.isfile(os.path.join(spectra_path, f))]
print("Running PPXF test on stack with {} spectra".format(len(spectra_files)))

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

t = clock()
stack = Stack(spectra_files)
print("- initialization: {}".format(clock()-t))
t = clock()
stack.prepare_spectra(wr, wlr, gs, wl=wl, dw=dw, dc=dc)
print("- spectra preparation: {}".format(clock()-t))
t = clock()
stack.determine_wavelength_range()
print("- wavelength range: {}".format(clock()-t))
t = clock()
stack.determine_weights()
print("- weights: {}".format(clock()-t))
t = clock()
stack.average()
print("- stacking: {}".format(clock()-t))
t = clock()
stack.jackknife()
print("- dispersion: {}".format(clock()-t))

x = np.arange(stack.N)
l = chunks(x, 5)

for j in l:
    axes = stack.plotSpectra(j, fl="flux_interp", wl="lam_interp", title="Spectra {} - {}".format(j[0]+1, j[-1]+1), show=False)
    for ax in axes:
        ax.plot(stack.wave, stack.flux, alpha=0.5)
        ax.plot(stack.wave, stack.dispersion, alpha=0.5)
    plt.show()
