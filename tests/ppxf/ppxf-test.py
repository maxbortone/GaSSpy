from time import clock
import sys, os
path = os.path.dirname(__file__).split('test')[0]
sys.path.append(path + "lib/")
from stacking import *

# import fits file and initialize stack
spectra_path = path + 'spectra_dustcorr/'
spectra_files = [os.path.join(spectra_path, f) for f in os.listdir(spectra_path) if os.path.isfile(os.path.join(spectra_path, f))]
stack = Stack(spectra_files)

# calculate stacked spectrum with masking and dust correction
wr = [4100, 4700]                   # wavelength range for normalization
gs = 1.0                            # grid spacing for interpolation
wl = np.array([5577, 6300, 6363])   # sky lines for masking
dw = 10                             # broadness of sky lines for masking
dc = True                           # flag for dust correction
stack.average(wr, gs, wl=wl, dw=dw, dc=dc)
# use Kroupa IMF
stack.ppxffit(temp='kb')
stack.plotFit('stack-fit-kb.png')
# refit without residuals in balmer lines
stack.ppxffit(temp='kb', refit=True)
stack.plotFit('stack-refit-kb.png')
