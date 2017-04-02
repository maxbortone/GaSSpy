from time import clock
import sys, os
path = os.path.dirname(__file__).split('test')[0]
sys.path.append(path + "lib/")
from stacking import *
from prettytable import PrettyTable

# import fits file and initialize stacks
sdss_path = path + 'SDSS_spectra/'
spectra_old = sdss_path + 'old/'
spectra_int = sdss_path + 'intermediate/'
spectra_yng = sdss_path + 'young/'

spectra_old_files = [os.path.join(spectra_old, f) for f in os.listdir(spectra_old) if os.path.isfile(os.path.join(spectra_old, f))]
spectra_int_files = [os.path.join(spectra_int, f) for f in os.listdir(spectra_int) if os.path.isfile(os.path.join(spectra_int, f))]
spectra_yng_files = [os.path.join(spectra_yng, f) for f in os.listdir(spectra_yng) if os.path.isfile(os.path.join(spectra_yng, f))]

stack_old = Stack(spectra_old_files, name="old")
stack_int = Stack(spectra_int_files, name="intermediate")
# TODO: look at individual spectra
stack_yng = Stack(spectra_yng_files, name="young")

# calculate stacked spectrum with masking and dust correction
wr = [4100, 4700]                   # wavelength range for normalization
gs = 1.0                            # grid spacing for interpolation
wl = np.array([5577, 6300, 6363])   # sky lines for masking
dw = 10                             # broadness of sky lines for masking
dc = True                           # flag for dust correction
stack_old.average(wr, gs, wl=wl, dw=dw, dc=dc)
stack_int.average(wr, gs, wl=wl, dw=dw, dc=dc)
stack_yng.average(wr, gs, wl=wl, dw=dw, dc=dc)

# calculate dispersion and bias
# TODO: debug runtime error: invalid value encountered in double_scalars
# in stacking.py:383 and stacking.py:408
stack_old.jackknife()
stack_old.correct()
stack_int.jackknife()
stack_int.correct()
stack_yng.jackknife()
stack_yng.correct()

# fit stacked spectra using ppxf with Kroupa IMF
stack_old.ppxffit(temp="kb")
stack_int.ppxffit(temp="kb")
stack_yng.ppxffit(temp="kb")

os.chdir(path + 'test/sdss/')

# print out the kinematics from ppxf
table = PrettyTable(["Population", "Vel", "Sigma", "h3", "h4"])
old_row = ["Old"]
old_row.extend(stack_old.pp.sol)
int_row = ["Intermediate"]
int_row.extend(stack_int.pp.sol)
yng_row = ["Young"]
yng_row.extend(stack_yng.pp.sol)
table.add_row(old_row)
table.add_row(int_row)
table.add_row(yng_row)
table_txt = table.get_string()
with open('output.txt','w') as file:
    file.write(table_txt)
    file.write('\n')

# plot stacked spectra
stack_old.plotStack('stack_old.png', title="Old")
stack_int.plotStack('stack_int.png', title="Intermediate")
stack_yng.plotStack('stack_yng.png', title="Young")

# plot ppxf fit
stack_old.plotFit('fit_old.png', title="Old")
stack_int.plotFit('fit_int.png', title="Intermediate")
stack_yng.plotFit('fit_yng.png', title="Young")

# print out the age, metallicity and mass to light estimates
# TODO: delog output age
table = PrettyTable(["Population", "Age", "Metallicity", "Mass to light"])
table.add_row(["Old", stack_old.mean_log_age, stack_old.mean_metal, stack_old.mass_to_light])
table.add_row(["Intermediate", stack_int.mean_log_age, stack_int.mean_metal, stack_int.mass_to_light])
table.add_row(["Young", stack_yng.mean_log_age, stack_yng.mean_metal, stack_yng.mass_to_light])
table_txt = table.get_string()
with open('output.txt','a') as file:
    file.write(table_txt)
