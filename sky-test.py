from stacking.stacking import *
from os import listdir
from os.path import isfile, join
from numpy import empty
from numpy.ma import where
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

spectra_path = './spectra'
spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]
stack = Stack(spectra_files)

def analyzePeak(sky, wl, lam, dlam, ax=None):
    a = lam-dlam
    b = lam+dlam
    wl_range = where((wl > a) & (wl < b))[0]
    sky_max = sky[wl_range].max()
    if ax is not None:
        ax.vlines([a, b], 0, 400, colors="r")
        if sky_max > 360:
            sky_y = 350
        else:
            sky_y = sky_max
        ax.text(b, sky_y, "%4d"%sky_max)
    return sky_max

# analyze sky lines at 5577, 6300 and 6363
## plot first 5 spectra
k = 5
f, axes = plt.subplots(k, 1, sharex=True)
for i in range(k):
    sp = stack.spectra[i]
    sky = sp.sky
    wl = 10**sp.loglam
    axes[i].plot(wl, sky, label="sky flux")
    analyzePeak(sky, wl, 5577, 10, ax=axes[i])
    analyzePeak(sky, wl, 6300, 10, ax=axes[i])
    analyzePeak(sky, wl, 6363, 10, ax=axes[i])
    axes[i].yaxis.set_major_locator(MultipleLocator(200.0))
    axes[i].set_ylim([0, 400])

# plt.show()
plt.savefig('sky-analysis.png')

## do complete analysis
peaks = empty((stack.N, 3))
for i in range(stack.N):
    sp = stack.spectra[i]
    sky = sp.sky
    wl = 10**sp.loglam
    sky_5577 = analyzePeak(sky, wl, 5577, 10)
    sky_6300 = analyzePeak(sky, wl, 6300, 10)
    sky_6363 = analyzePeak(sky, wl, 6363, 10)
    peaks[i] = [sky_5577, sky_6300, sky_6363]

f, axes = plt.subplots(3, 1, sharex=True)
wls = [5577, 6300, 6363]
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
for i in range(3):
    mu = peaks[:, i].mean()
    std = peaks[:, i].std()
    axes[i].hist(peaks[:, i], bins=10)
    axes[i].set_title("Sky flux distribution at %4d angstrom"%wls[i])
    textstr = "$\mu = %.2f$\n$\sigma = %.2f$"%(mu, std)
    axes[i].text(0.75, 0.95, textstr, transform=axes[i].transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
# plt.show()
plt.savefig('sky-histo.png')
