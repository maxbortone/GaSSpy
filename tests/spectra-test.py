from astroquery.sdss import SDSS
from astropy import coordinates as coords
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# setup latex in matplotlib
# see:
#   - http://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
#   - http://matplotlib.org/users/customizing.html
mpl.rcParams['mathtext.fontset'] = 'stixsans'
mpl.rcParams['font.family'] = 'sans'
mpl.rcParams['font.serif'] = 'STIXGeneral'
mpl.rcParams['text.usetex'] =True
mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',     # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',     # ...this to force siunitx to actually use your fonts
       r'\usepackage{sans}',        # set the normal font here
       r'\usepackage{sansmath}',    # load up the sansmath so that math -> helvet
       r'\sansmath',                # <- tricky! -- gotta actually tell tex to use!
       r'\DeclareSIUnit\erg{erg}'
]

pos = coords.SkyCoord('0h8m05.63s +14d50m23.3s', frame='icrs')
xid = SDSS.query_region(pos, spectro=True)
sp = SDSS.get_spectra(matches=xid)
hdulist = sp[0]
hdulist.info()
header = hdulist[1].header
print(repr(header))

tbdata = hdulist[1].data
cols = hdulist[1].columns
flux = tbdata['flux']
sky = tbdata['sky']
model = tbdata['model']
ivar = tbdata['ivar']
loglam = tbdata['loglam']

f, axes = plt.subplots(1, 1, figsize=(7.0, 5.0))

axes.plot(10**loglam, flux, label="coadded calibrated flux")
axes.plot(10**loglam, sky, label="subtracted sky flux")
axes.plot(10**loglam, model, label="best model fit")
axes.plot(10**loglam, ivar, label="noise")


axes.set_xlim([3700, 9300])
axes.set_ylim([0, 180])

# axes.set_ylabel(r'$F_{\lambda}\/\/\/[10^{-17}\/\mathrm{erg}\/\mathrm{s}^{-1}\/\mathrm{cm}^{-2}\/\mathrm{\AA}^{-1}]$')
axes.set_ylabel(r"$F_{\lambda}$ [\SI{e-17}{\erg\per\second\per\square\centi\meter\per\angstrom}]")
axes.set_xlabel(r"$\lambda$ [\si{\angstrom}]")

# axes.xaxis.set_major_locator(MultipleLocator(50))
# axes.xaxis.set_minor_locator(MultipleLocator(10))
axes.yaxis.set_major_locator(MultipleLocator(500))
axes.yaxis.set_minor_locator(MultipleLocator(100))
plt.setp(axes.yaxis.get_ticklines(), 'markeredgewidth', 1.5)
# plt.setp(axes.xaxis.get_ticklines(), 'markeredgewidth', 1.5)

axes.legend(loc="best", frameon=False)

plt.show()
# plt.savefig("spectra-test.png")
