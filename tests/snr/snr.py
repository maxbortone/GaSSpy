from time import clock
import sys, os
path = os.path.dirname(__file__).split('test')[0]
sys.path.append(path + "lib/")
import numpy as np
from stack import Stack
from spectrum import Spectrum
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# import fits file and initialize stack
# spectra_path = path + 'spectra_dustcorr/'
# spectra_path = path + 'SDSS_spectra/young'
spectra_path = path + 'SDSS_spectra/intermediate'
# spectra_path = path + 'SDSS_spectra/old'

spectra_files = [os.path.join(spectra_path, f) for f in os.listdir(spectra_path) if os.path.isfile(os.path.join(spectra_path, f))]
N = len(spectra_files)

# wavelength regions for signal-to-noise analysis
wlr = np.array([[4500, 4700],
                [5400, 5600],
                [6000, 6200],
                [6800, 7000]])
colors = ['b', 'g', 'r', 'y']
M = wlr.shape[0]
SNRs = np.empty((N, M))
signals = np.empty(N)
noises = np.empty(N)

# calculate stacked spectrum with masking and dust correction
wr = [4100, 4700]                   # wavelength range for normalization
gs = 1.0                            # grid spacing for interpolation
wl = np.array([5577, 6300, 6363])   # sky lines for masking
dw = 3                              # broadness of sky lines for masking
dc = True                           # flag for dust correction
stack = Stack(spectra_files)
stack.prepare_spectra(wr, wlr, gs, wl=wl, dw=dw, dc=dc)
stack.determine_weights()
stack.average()
SNRs_stack = stack.signaltonoise(wlr, flag=True)
f, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
for i in range(N):
    sp = stack.spectra[i]
    SNRs[i, :] = sp.signaltonoise(wlr, flag=True)
    signals[i] = sp.S
    noises[i] = sp.N
    x = (i+1)*np.ones(M)
    ax.scatter(x, SNRs[i, :], c=colors)
    ax.scatter(x[0], sp.SNR, c='k', marker='_')
    ax.text(x[0]+0.2, sp.SNR, s="{:.2f}".format(sp.SNR))
lines = []
for i in range(M):
    line = mlines.Line2D([], [], color=colors[i], markersize=15, label="{} - {}".format(wlr[i][0], wlr[i][1]))
    lines.append(line)
ax.set_xlabel("spectra")
ax.set_ylabel("SNR")
ax.legend(handles=lines, loc="upper right", frameon=False)
ax.scatter((N/2)*np.ones(M), SNRs_stack, c=colors, marker='v')
ax.scatter((N/2), stack.SNR, c='k', marker='*')
ax.text((N/2)+0.2, stack.SNR, s="{:.2f}".format(stack.SNR))

# weights = stack.weights / sum(stack.weights)
S_approx = sum(stack.weights*signals)
N_approx = np.sqrt(sum(stack.weights**2*(noises**2)))
SNR_approx = S_approx / N_approx
ax.scatter((N/2), SNR_approx, c='m', marker='*')
ax.text((N/2)-1.2, SNR_approx, s="{:.2f}".format(SNR_approx))

print("SNR approximation = {:.2f}".format(SNR_approx))
print("SNR of the stack = {:.2f}".format(stack.SNR))
print("ratio = {:.2f}".format(stack.SNR / SNR_approx))

plt.show()
