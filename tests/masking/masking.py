from time import clock
import sys, os
path = os.path.dirname(__file__).split('test')[0]
sys.path.append(path + "lib/")
import numpy as np
from stacking import *
import matplotlib.colors as colors

# import fits file and initialize stack
# spectra_path = path + 'spectra_dustcorr/'
spectra_path = path + 'SDSS_spectra/young'
# spectra_path = path + 'SDSS_spectra/intermediate'
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

t = clock()
stack.average(wr, gs, wl=wl, dw=dw, dc=dc)
print("- stacking: {}".format(clock()-t))

lam1 = np.array(stack.wave)
galaxy1 = np.array(stack.flux)

t = clock()
stack.correct()
print("- bias correction: {}".format(clock()-t))

lam2 = np.array(stack.wave)
galaxy2 = np.array(stack.flux)
contribs = stack.contributions/stack.N

# z = 0
# for i in range(stack.N):
#     sp = stack.spectra[i]
#     z += 1+sp.z[0]
# z /= stack.N

cm = plt.get_cmap('jet')
cn  = colors.Normalize(vmin=contribs.min(), vmax=contribs.max())

f, ax0 = plt.subplots(1, 1, figsize=(7.0, 5.0))
ax0.plot(lam1, galaxy1, label="stack")
ax0.plot(lam2, galaxy2, label="bias corr")
ax0.plot(lam2, contribs, label="contributions")
# mm = ax0.scatter(lam2, galaxy2, s=10, c=contribs, cmap=cm, norm=cn)
ax0.legend(loc="lower right", frameon=False)
ax0.text(0.05, 0.95, "Stack size: {}".format(stack.N), transform=ax0.transAxes)
# ax0.vlines(wl, ax0.get_ylim()[0], ax0.get_ylim()[1])
# ax0.vlines(wl[0]+z, ax0.get_ylim()[0], ax0.get_ylim()[1], color='r')
# f.colorbar(mm)
plt.show()
