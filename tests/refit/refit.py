from time import clock
import sys, os
path = os.path.dirname(__file__).split('test')[0]
sys.path.append(path + "lib/")
import numpy as np
from stack import Stack
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import lineid_plot

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',
       r'\DeclareSIUnit\ergs{ergs}',
       r'\sisetup{per-mode=symbol}'
]

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
print("Running refit test on stack with {} spectra".format(len(spectra_files)))

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

# emission lines
emlines = []
emlabels = []
for el in stack.emlines:
    if isinstance(el, tuple):
        emlines.append(el[1])
        emlabels.append(el[0])
    elif isinstance(el, list):
        for l in el:
            emlines.append(l[1])
            emlabels.append(l[0])
for i in range(len(emlabels)):
    label = emlabels[i]
    if "_" in label:
        parts = label.split("_")
        label = '$' + parts[0] + '_{\\' + parts[1] + '}$'
        emlabels[i] = label
    else:
        label = '$' + label + '$'
        emlabels[i] = label

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

f, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.0, 10.0), sharex=True)
f.suptitle('Emission line fill in correction')
handle_stack, = ax0.plot(lam1, galaxy1, label="stack")
handle_fit, = ax0.plot(lam1, bestfit1, label="fit")
handle_residual, = ax0.plot(lam1, residual1, label="residual")
ax1.plot(lam2, galaxy2, label="stack")
ax1.plot(lam2, bestfit2, label="fit")
ax1.plot(lam2, residual2, label="residual")
ax1.plot(lam2, bestfit2-bestfit1, label="after - before")
lineid_plot.plot_line_ids(stack.wave, stack.flux, emlines, emlabels, ax=ax0, max_iter=300, extend=False, add_label_to_artists=False)
for i, line in enumerate(emlines):
    ax0.axvline(line, color='k', linewidth=1, linestyle='--')
    ax1.axvline(line, color='k', linewidth=1, linestyle='--')
for (key, val) in gaussians.items():
    ax0.plot(val['gx'], val['gs'], 'lightblue')
handle_gaussian = mlines.Line2D([], [], color='lightblue', markersize=15, label="gaussian")
ax0.set_ylabel(r"flux [\SI{e-17}{\ergs\per\second\per\square\centi\meter\per\angstrom}]")
ax1.set_ylabel(r"flux [\SI{e-17}{\ergs\per\second\per\square\centi\meter\per\angstrom}]")
ax1.set_xlabel(r"wavelength [\si{\angstrom}]")
ax0.set_title('before', loc='right')
ax0.legend(handles=[handle_stack, handle_fit, handle_residual, handle_gaussian], loc='best', frameon=False)
ax1.set_title('after', loc='right')
ax1.legend(loc='best', frameon=False)
plt.show()
