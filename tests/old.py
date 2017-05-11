###############################################################################
# TESTS
###############################################################################

def main1():
    from time import clock
    # load all fits filenames
    spectra_path = './spectra_dustcorr'
    spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]
    wr = [4100, 4700]
    gs = 1.0
    wl = np.array([5577, 6300, 6363])
    dw = 10
    dc = True
    tp = 'kb'
    t = clock()
    stack = Stack(spectra_files)
    print('Elapsed time in stack initialization: %.2f s' % (clock() - t))
    t = clock()
    lam, flux, noise, contrs = stack.average(wr, gs, wl=wl, dw=dw, dc=dc)
    print('Elapsed time in stacking: %.2f s' % (clock() - t))
    t = clock()
    disp = stack.jackknife()
    print('Elapsed time in jackknife: %.2f s' % (clock() - t))
    t = clock()
    corr = stack.correct()
    print('Elapsed time in correct: %.2f s' % (clock() - t))
    t = clock()
    stack.ppxffit(temp=tp)
    print('Elapsed time in ppxffit: %.2f s' % (clock() - t))
    sigma_A, sigma_Z = error(stack, wr, gs, wl, dw, dc, tp)
    print('Elapsed time in error: %.2f s' % (clock() - t))

    print('Mean log age: %f +- %f Gyr' % (stack.mean_log_age, sigma_A))
    print('Mean log age: %f +- %f ' % (stack.mean_metal, sigma_Z))

    # indices = range(5)
    # axes1 = stack.plotSpectra(indices, fl=['noise_norm', 'flux_norm'], wl='loglam_dered', show=False, filename='spectra-normed.png')

    f, axes = plt.subplots(1, 1, figsize=(16.0, 9.0))
    axes.plot(lam, flux, label="stacked flux")
    axes.plot(lam, noise, label="noise")
    axes.plot(lam, disp, label="dispersion")
    axes.plot(lam, corr, label="correction")
    axes.legend(loc="best", frameon=False)
    plt.show()
    # plt.savefig('spectra-stacked.png')

def main2():
    spectra_path = './spectra'
    spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]
    stack = Stack(spectra_files)
    k = len(spectra_files)

    wl = np.array([5577, 6300, 6363])
    dw = 10
    lam, flux, noise, contr = stack.average([4100, 4700], 1.0)
    lam_masked, flux_masked, noise_masked, contr_masked = stack.average([4100, 4700], 1.0, wl=wl, dw=dw)

    f, axes = plt.subplots(2, 1, sharex=True, figsize=(16.0, 9.0))
    ln = [wl-dw,wl+dw]
    axes[0].plot(lam, flux, label="stacked flux")
    axes[0].plot(lam_masked, flux_masked, label="with masking")
    axes[0].plot(lam_masked, noise_masked, label="noise")
    axes[0].vlines(wl, 0, flux.max(), label="sky emission lines")
    # axes[0].vlines(ln, 0, flux.max(), linestyle="dotted")
    axes[0].legend(loc="best", frameon=False, fontsize="12")
    axes[1].vlines(wl, 0, k+2)
    # axes[1].vlines(ln, 0, k+2, linestyle="dotted")
    axes[1].plot(lam, contr_masked, label="contributions")
    axes[1].legend(loc="best", frameon=False, fontsize="12")
    # plt.show()
    plt.savefig('spectra-masking.png')

def main3():
    spectra_path = './spectra_dustcorr'
    spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]
    stack = Stack(spectra_files)
    k = len(spectra_files)

    wl = np.array([5577, 6300, 6363])
    dw = 10
    lam, flux, noise, contr = stack.average([4100, 4700], 1.0, dc=True)
    lam_masked, flux_masked, noise_masked, contr_masked = stack.average([4100, 4700], 1.0, wl=wl, dw=dw, dc=True)

    f, axes = plt.subplots(2, 1, sharex=True, figsize=(16.0, 9.0))
    ln = [wl-dw,wl+dw]
    axes[0].plot(lam, flux, label="stacked flux")
    axes[0].plot(lam_masked, flux_masked, label="with masking")
    axes[0].plot(lam_masked, noise_masked, label="noise")
    axes[0].vlines(wl, 0, flux.max(), label="sky emission lines")
    # axes[0].vlines(ln, 0, flux.max(), linestyle="dotted")
    axes[0].legend(loc="best", frameon=False, fontsize="12")
    axes[1].vlines(wl, 0, k+2)
    # axes[1].vlines(ln, 0, k+2, linestyle="dotted")
    axes[1].plot(lam, contr_masked, label="contributions")
    axes[1].legend(loc="best", frameon=False, fontsize="12")
    # plt.show()
    plt.savefig('spectra-dustcorr-masking.png')

def main4():
    spectra_path = './spectra_dustcorr'
    spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]
    stack = Stack(spectra_files)

    wl = np.array([5577, 6300, 6363])
    dw = 10
    lam, flux, noise, contr = stack.average([4100, 4700], 1.0, wl=wl, dw=dw)
    lam_dc, flux_dc, noise_dc, contr_dc = stack.average([4100, 4700], 1.0, wl=wl, dw=dw, dc=True)

    f, axes = plt.subplots(1, 1, sharex=True, figsize=(16.0, 9.0))
    axes.plot(lam, flux, label="stacked flux")
    axes.plot(lam_dc, flux_dc, label="with dust correction")
    axes.plot(lam, abs(flux_dc-flux), label="difference")
    axes.legend(loc="best", frameon=False, fontsize="12")
    # plt.show()
    plt.savefig('spectra-dustcorr.png')

def main5():
    spectra_path = './spectra_dustcorr'
    spectra_files = [join(spectra_path, f) for f in listdir(spectra_path) if isfile(join(spectra_path, f))]

    sp = Spectrum(spectra_files[0])

    f, axes = plt.subplots(1, 1, sharex=True, figsize=(16.0, 9.0))
    axes.plot(10**sp.loglam, sp.flux, label="flux")
    axes.plot(10**sp.loglam, sp.flux_dustcorr, label="with dust correction")
    axes.plot(10**sp.loglam, abs(sp.flux_dustcorr-sp.flux), label="difference")
    axes.legend(loc="best", frameon=False, fontsize="12")
    # plt.show()
    plt.savefig('spectra-comp.png')
