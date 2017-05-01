from time import clock
import sys, os
path = os.path.dirname(__file__).split('test')[0]
sys.path.append(path + "lib/")
import numpy as np
from stacking import *

#######################################################################
# TEST: ppxf fitting and refitting
#   1. stacking --> ppxf fit --> bestfit and residual
#   2. stack - residual in Balmer lines --> improved stack
#   3. ppxf refit without masking in Balmer lines --> improved bestfit
#######################################################################

# import fits file and initialize stack
# spectra_path = path + 'spectra_dustcorr/'
# spectra_path = path + 'SDSS_spectra/young'
spectra_path = path + 'SDSS_spectra/intermediate'
# spectra_path = path + 'SDSS_spectra/old'

spectra_files = [os.path.join(spectra_path, f) for f in os.listdir(spectra_path) if os.path.isfile(os.path.join(spectra_path, f))]
t = clock()
stack = Stack(spectra_files)
print("- stack initialization: {}".format(clock()-t))

# calculate stacked spectrum with masking and dust correction
wr = [4100, 4700]                   # wavelength range for normalization
gs = 1.0                            # grid spacing for interpolation
wl = np.array([5577, 6300, 6363])   # sky lines for masking
dw = 10                             # broadness of sky lines for masking
dc = True                           # flag for dust correction
tp = 'kb'                           # Kroupa IMF
balmer_lines = [4101.76, 4340.47, 4861.33]
balmer_names = [r"$H_{\delta}$", r"$H_{\gamma}$", r"$H_{\beta}$"]

t = clock()
stack.average(wr, gs, wl=wl, dw=dw, dc=dc)
print("- stacking: {}".format(clock()-t))
t = clock()
stack.ppxffit(temp=tp)
print("- ppxf fit: {}".format(clock()-t))

galaxy1 = np.array(stack.pp.galaxy)
bestfit1 = np.array(stack.pp.bestfit)
residual1 = np.array(stack.residual)
lam1 = np.array(stack.pp.lam)

t = clock()
stack.ppxffit(temp=tp, refit=True)
print("- ppxf refit: {}".format(clock()-t))

galaxy2 = np.array(stack.pp.galaxy)
bestfit2 = np.array(stack.pp.bestfit)
residual2 = np.array(stack.residual)
lam2 = np.array(stack.pp.lam)
gaussians = stack.gaussians

f, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.0, 10.0))
f.suptitle('Emission line fill in correction')
ax0.plot(lam1, galaxy1, label="stack")
ax0.plot(lam1, bestfit1, label="fit")
ax0.plot(lam1, residual1, label="residual")
for i, line in enumerate(balmer_lines):
    ax0.axvline(line, color='r', linewidth=1, linestyle='--')
    ax0.text(line+20, ax0.get_ylim()[1]-0.1, balmer_names[i], color='r')
for i in range(len(gaussians)):
    wv = gaussians[i][0]
    gs = gaussians[i][1]
    ax0.plot(wv, gs)
    ax0.text(wv[-1], gs.max(), "gaussian", color='r')
ax0.set_title('before')
ax0.legend(loc='best', frameon=False)

ax1.plot(lam2, galaxy2, label="stack")
ax1.plot(lam2, bestfit2, label="fit")
ax1.plot(lam2, residual2, label="residual")
ax1.plot(lam2, bestfit2-bestfit1, label="diff")
for i, line in enumerate(balmer_lines):
    ax1.axvline(line, color='r', linewidth=1, linestyle='--')
    ax1.text(line+20, ax0.get_ylim()[1]-0.1, balmer_names[i], color='r')
ax1.set_title('after')
ax1.legend(loc='best', frameon=False)
plt.show()
