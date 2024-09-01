import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import SynthObs
import SynthObs.Morph
from SynthObs.SED import models
from SynthObs.Morph import measure
from SynthObs.Morph import images
from SynthObs.Morph import PSF
import FLARE.filters
from matplotlib.patches import Circle
import pandas as pd
from synphot import etau_madau
from mpl_toolkits.axes_grid1 import make_axes_locatable
import make_background
from photutils import aperture_photometry
from photutils import CircularAperture
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian2D

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def plot_PSF(LPSF, super_samp, f, noise, background):
    filt_str = (f.split('.')[-1])

    img = images.observed(f, cosmo, z, target_width_arcsec=width, smoothing=False, verbose=True, PSF=PSFs[f],
                          super_sampling=super_samp).particle(np.array([0.]), np.array([0.]), np.array([LPSF]),
                                                              centre=[0, 0, 0])

    if background:
        # create background image object (cutoutwidth in pixels)
        ap_sig = 10 # 10 sigma observation
        background_object = make_background.Background(zeropoint=zeropoint, pixel_scale=pixel_scale / super_samp,
                                                       aperture_f_limit=aperture_f_limit, aperture_significance=ap_sig,
                                                       aperture_radius=aperture_radius, verbose=True)
        img_bkg = background_object.create_background_image(Npixels * super_samp)



        img_bkg_data = img_bkg.bkg * nJy_to_es
        bkg_sigma = background_object.pixel.noise_es * np.ones_like(img_bkg.bkg)

    mask = create_circular_mask(Npixels * super_samp, Npixels * super_samp, center=None,
                                radius=np.floor(Npixels * super_samp / 2))
    img_data = img.super.data * nJy_to_es


    if noise:
        # Add shot noise to full noise
        full_img = img_data * exp_time
        full_img[full_img < 0] = 0
        noisy_full_img = np.random.poisson(full_img)
        img_data = noisy_full_img / exp_time

    if background:
        y, x = np.mgrid[0:len(img_bkg.bkg), 0:len(img_bkg.bkg)]
        gauss = Gaussian2D(np.max(img_data) / 5000, len(img_bkg.bkg) / 2 - 1.5, len(img_bkg.bkg) / 2 - 1.5, 1.5, 1.5)(x,
                                                                                                                      y)
        print('Center loc: ', len(img_bkg.bkg) / 2 - 1.5)
        ivm = 1 / ((bkg_sigma * super_samp) ** 2 + (gauss))
    else: # IVM just 0s
        ivm = np.full_like(img_data, 0)


    hdu = fits.PrimaryHDU(mask * img_data) # saving main psf
    hdu_ivm = fits.PrimaryHDU(mask * ivm) # saving IVM for psfmc


    if flux_fact != 20:
        hdu.writeto(
            '/fred/oz183/sberger/paper_1_bluetides/data/sci_PSF_JWST_{}_{}'.format(filt_str, int(flux_fact)) + '.fits',
            overwrite=True)
        hdu_ivm.writeto(
            '/fred/oz183/sberger/paper_1_bluetides/data/ivm_PSF_JWST_{}_{}'.format(filt_str, int(flux_fact)) + '.fits',
            overwrite=True)
    if background and noise:
        hdu.writeto(
            '/fred/oz183/sberger/paper_1_bluetides/data/sci_PSF_JWST_{}_SN_{}ss_{}s.fits'.format(filt_str,
                                                                                                     super_samp,
                                                                                                    exp_time),
            overwrite=True)
        hdu_ivm.writeto(
            '/fred/oz183/sberger/paper_1_bluetides/data/ivm_PSF_JWST_{}_SN_{}ss_{}s.fits'.format(filt_str,
                                                                                                    super_samp,
                                                                                                    exp_time),
            overwrite=True)
    else:
        psf_name = '/fred/oz183/sberger/paper_1_bluetides/data/NOISELESS_sci_PSF_JWST_{}_SN_{}ss_{}s.fits'.format(filt_str, super_samp, exp_time)
        ivm_psf_name = '/fred/oz183/sberger/paper_1_bluetides/data/NOISELESS_ivm_PSF_JWST_{}_SN_{}ss_{}s.fits'.format(filt_str, super_samp, exp_time)

        print(psf_name)
        print(ivm_psf_name)
        hdu.writeto(psf_name,
            overwrite=True)
        hdu_ivm.writeto(ivm_psf_name,
            overwrite=True)
        print("Noiseless PSF created!")

    return


if __name__ == '__main__':
    # Setup
    cosmo = FLARE.default_cosmo()
    z = 6.560000154967445 # exact BlueTides redshift
    if '--noise' in sys.argv:
        noise = True
    else:
        noise = False

    if '--background' in sys.argv:
        background = True
    else:
        background = False

    filter = sys.argv[1]
    super_samp = int(sys.argv[2])
    exp_time = int(sys.argv[3]) #3100  # 10000

    model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300') # DEFINE SED GRID, I don't think this matters since we're just grabbing wavelengths

    F = FLARE.filters.add_filters([filter], new_lam=model.lam * (1. + z))
    PSFs = PSF.Webb([filter], resampling_factor=5)  # creates a dictionary of instances of the webbPSF class

    width = 3  # 8.33 #size of cutout in ''  #MUST RESULT IN EVEN NUMBER OF PIXELS
    FOV = width / cosmo.arcsec_per_kpc_proper(z).value  # size of cutout in kpc
    smoothing = None  # ('adaptive',60)

    pixel_scale = FLARE.filters.pixel_scale[filter]  # arcsec/pixel (for NIRCam SW)
    Npixels = int(width / pixel_scale)  # 20#width of image / resolution
    # background setup
    aperture_radius = 2.5 * pixel_scale  # aperture radius in arcsec
    zeropoint = 25.946
    nJy_to_es = 1E-9 * 10 ** (0.4 * (zeropoint - 8.9))
    if exp_time == 10000:
        aperture_flux_limits = {'JWST.NIRCAM.F090W': 15.3, 'JWST.NIRCAM.F115W': 13.2,
                                'JWST.NIRCAM.F150W': 10.6, 'JWST.NIRCAM.F200W': 9.1, 'JWST.NIRCAM.F277W': 14.3,
                                'JWST.NIRCAM.F356W': 12.1, 'JWST.NIRCAM.F444W': 23.6, 'JWST.MIRI.F560W': 130,
                                'JWST.MIRI.F770W': 240}  # sensitivity at 10ks in nJy, 10 sigma
        aperture_f_limit = aperture_flux_limits[filter]
    else: # other exposure times specified in the dictionary
        if filter == 'JWST.NIRCAM.F200W':
            aperture_flux_limits = {1000: 44.9, 2500: 19.9, 5000: 13.1,
                                    10000: 9.1}  # 10 sigma limits for F200W, 1ks, 5ks, 10ks
        elif filter == 'JWST.NIRCAM.F150W':
            aperture_flux_limits = {3100: 17.15}  # UPDATED-Sabrina
        elif filter == 'JWST.NIRCAM.F115W':
            aperture_flux_limits = {5000: 18.1}  # 10 sigma limits for F150W, 1ks, 5ks, 10ks
        elif filter == 'JWST.NIRCAM.F277W':
            aperture_flux_limits = {5000: 19.1}  # 10 sigma limits for F150W, 1ks, 5ks, 10ks
        elif filter == 'JWST.NIRCAM.F356W':
            aperture_flux_limits = {3100: 21.73}  # UPDATED-Sabrina
        elif filter == 'JWST.NIRCAM.F444W':
            aperture_flux_limits = {5000: 24.7}  # 10 sigma limits for F150W, 1ks, 5ks, 10ks
        else:
            print("ERR, can't have not specified filter with non-10ks exposure time")
        aperture_f_limit = aperture_flux_limits[exp_time]
    print('Aperture flux limit ', aperture_f_limit, 'Exp time ', exp_time)



    # FPSF=13578*12 #12x BT model SDSS quasar brightness, from difference between PSF and quasar for SDSS-J0203
    flux_fact = 20
    FPSF = 6e5  # 16941.5654116655 * flux_fact
    plot_PSF(FPSF, super_samp, filter, noise, background)


