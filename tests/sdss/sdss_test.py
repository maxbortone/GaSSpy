from time import clock
import sys, os
path = os.path.dirname(__file__).split('test')[0]
sys.path.append(path + "lib/")
import numpy as np
from stack import Stack
from prettytable import PrettyTable
import matplotlib.pyplot as plt

plt.style.use('fivezerosix')

# import fits file and initialize stacks
sdss_path = path + 'SDSS_spectra/'
spectra_old = sdss_path + 'old/'
spectra_int = sdss_path + 'intermediate/'
spectra_yng = sdss_path + 'young/'

spectra_old_files = [os.path.join(spectra_old, f) for f in os.listdir(spectra_old) if os.path.isfile(os.path.join(spectra_old, f))]
spectra_int_files = [os.path.join(spectra_int, f) for f in os.listdir(spectra_int) if os.path.isfile(os.path.join(spectra_int, f))]
spectra_yng_files = [os.path.join(spectra_yng, f) for f in os.listdir(spectra_yng) if os.path.isfile(os.path.join(spectra_yng, f))]
print("Running SDSS test on 3 stacks:\n\t old: {} spectra\n\t intermediate: {} spectra\n\t young: {} spectra"
      .format(len(spectra_old_files), len(spectra_int_files), len(spectra_yng_files)))

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
tp = 'default'                      # Kroupa IMF

t = clock()
stack_old = Stack(spectra_old_files, name="old")
stack_int = Stack(spectra_int_files, name="intermediate")
stack_yng = Stack(spectra_yng_files, name="young")
print("- initialization: {}".format(clock()-t))

t = clock()
stack_old.prepare_spectra(wr, wlr, gs, wl=wl, dw=dw, dc=dc)
stack_int.prepare_spectra(wr, wlr, gs, wl=wl, dw=dw, dc=dc)
stack_yng.prepare_spectra(wr, wlr, gs, wl=wl, dw=dw, dc=dc)
print("- spectra preparation: {}".format(clock()-t))

t = clock()
stack_old.determine_wavelength_range()
stack_int.determine_wavelength_range()
stack_yng.determine_wavelength_range()
print("- wavelength range: {}".format(clock()-t))

t = clock()
stack_old.determine_weights()
stack_int.determine_weights()
stack_yng.determine_weights()
print("- weights: {}".format(clock()-t))

t = clock()
stack_old.average()
stack_int.average()
stack_yng.average()
print("- stacking: {}".format(clock()-t))

t = clock()
stack_old.jackknife()
stack_int.jackknife()
stack_yng.jackknife()
print("- dispersion: {}".format(clock()-t))

t = clock()
stack_old.ppxffit(temp=tp)
stack_int.ppxffit(temp=tp)
stack_yng.ppxffit(temp=tp)
print("- ppxf fit: {}".format(clock()-t))

os.chdir(path + 'tests/sdss/')

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
table1_txt = table.get_string()

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
table2_txt = table.get_string()
with open('output_{}.txt'.format(tp),'w') as file:
    file.write(table1_txt)
    file.write('\n')
    file.write(table2_txt)
