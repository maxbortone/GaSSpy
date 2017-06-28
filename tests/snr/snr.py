from time import clock
import sys, os
path = os.path.dirname(__file__).split('test')[0]
sys.path.append(path + "lib/")
import numpy as np
from stack import Stack
from spectrum import Spectrum
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.style.use('fivezerosix')

# import fits file and initialize stack
# spectra_path = path + 'spectra_dustcorr/'
# spectra_path = path + 'SDSS_spectra/young'
# spectra_path = path + 'SDSS_spectra/intermediate'
# spectra_path = path + 'SDSS_spectra/old'
spectra_path = path + 'SDSS_spectra/bin_15'

spectra_files = [os.path.join(spectra_path, f) for f in os.listdir(spectra_path) if os.path.isfile(os.path.join(spectra_path, f))]
print("Running signal-to-noise test on stack with {} spectra".format(len(spectra_files)))

N = len(spectra_files)

# wavelength regions for signal-to-noise analysis
wlr = np.array([[4500, 4700],
                [5400, 5600],
                [6000, 6200],
                [6800, 7000]])
colors = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f']
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

SNRs_stack = stack.signaltonoise(wlr, flag=True)
f, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
for i in range(N):
    sp = stack.spectra[i]
    SNRs[i, :] = sp.signaltonoise(wlr, flag=True)
    signals[i] = sp.S
    noises[i] = sp.N
    x = (i+1)*np.ones(M)
    ax.scatter(x, SNRs[i, :], c=colors, marker='_')
    ax.scatter(x[0], sp.SNR, c='k', marker='_')
    # ax.text(x[0], SNRs[i, -1]+4, s="{:.2f}".format(sp.SNR), ha='center')
lines = []
for i in range(M):
    line = mlines.Line2D([], [], color=colors[i], markersize=15, label="{} - {}".format(wlr[i][0], wlr[i][1]))
    lines.append(line)
ax.set_xlabel("spectra")
ax.set_ylabel("SNR")
ax.scatter((N/2)*np.ones(M), SNRs_stack, c=colors, marker='v')
ax.scatter((N/2), stack.SNR, c='k', marker='v')
# ax.text((N/2)+0.2, stack.SNR, s="{:.2f}".format(stack.SNR))

# weights = stack.weights / sum(stack.weights
weights = np.zeros(stack.N)
for i in range(stack.N):
    weights[i] = np.median(stack.weights[i, :])
S_approx = sum(weights*signals)
N_approx = np.sqrt(sum(weights**2*(noises**2)))
SNR_approx = S_approx / N_approx
ax.scatter((N/2), SNR_approx, c='m', marker='*')
# ax.text((N/2)-1.2, SNR_approx, s="{:.2f}".format(SNR_approx))

line_median = mlines.Line2D([], [], color='k', markersize=15, label="median")
line_stack = mlines.Line2D([], [], color='k', linestyle='None', marker='v', markersize=10, label="stack")
line_approx = mlines.Line2D([], [], color='m', linestyle='None', marker='*', markersize=10, label="expected")
lines.extend([line_median, line_stack, line_approx])
ax.xaxis.set_ticks(np.arange(0, stack.N+1, 2))
ax.legend(handles=lines, loc="upper right", frameon=False)

print("SNR approximation = {:.2f}".format(SNR_approx))
print("SNR of the stack = {:.2f}".format(stack.SNR))
print("ratio = {:.2f}".format(stack.SNR / SNR_approx))

plt.show()
