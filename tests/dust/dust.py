from time import clock
import sys, os
path = os.path.dirname(__file__).split('test')[0]
sys.path.append(path + "lib/")
import numpy as np
from stack import Stack
import matplotlib.pyplot as plt

plt.style.use('fivezerosix')

# import fits file and initialize stack
# spectra_path = path + 'spectra_dustcorr/'
# spectra_path = path + 'SDSS_spectra/young'
# spectra_path = path + 'SDSS_spectra/intermediate'
# spectra_path = path + 'SDSS_spectra/old'
spectra_path = path + 'SDSS_spectra/bin_15'

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
stack.ppxffit(temp=tp)
print("- ppxf fit: {}".format(clock()-t))

galaxy1 = np.array(stack.flux)

t = clock()
stack.fit_residual()
print("- fit residual: {}".format(clock()-t))
t = clock()
stack.set_emlines_flux()
print("- set emission lines flux: {}".format(clock()-t))
t = clock()
stack.correct_dust_attenuation()
print("- correct dust attenuation: {}".format(clock()-t))

galaxy2 = np.array(stack.flux)

f, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
ax.plot(stack.wave, galaxy1, label="stacked spectrum")
ax.plot(stack.wave, galaxy2, label="with dust attenuation correction")
ax.set_ylabel(r"flux [\SI{e-17}{\ergs\per\second\per\square\centi\meter\per\angstrom}]")
ax.set_xlabel(r"wavelength [\si{\angstrom}]")
ax.legend(loc="best")
plt.show()
