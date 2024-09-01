from numpy import array
import numpy as np
from psfMC.ModelComponents import *
from psfMC.distributions import Normal, Uniform
from astropy.io import fits

filter_type = "{{ filter_type }}"
zp = {{ zp }}
direc_fits = "{{ direc_fits }}"
direc_ivm_fits = "{{ direc_ivm_fits }}"

fits_name = "{{ fits_name }}"
psf_name = "{{ psf_name }}"
ivm_psf_name = "{{ ivm_psf_name }}"
ivm_fits_name = "{{ ivm_fits_name }}"
center_x = {{ center_x }}
center_y = {{ center_y }}
mag_quasar = {{ mag_quasar }}
# mag_galaxy = {{ mag_galaxy }}
exp_time = {{ exp_time }}

Configuration(obs_file=direc_fits + fits_name,
              obsivm_file=direc_ivm_fits + ivm_fits_name,
              psf_files=f'/fred/oz183/sberger/paper_2_obs_bias/psf_fits_files/{psf_name}',
              psfivm_files=f'/fred/oz183/sberger/paper_2_obs_bias/psf_fits_files/{ivm_psf_name}',
              mag_zeropoint=zp)
# psf_files = f'../data/sci_PSF_JWST_{filter_type}_SN_{exp_time}s.fits',
# psfivm_files = f'../data/ivm_PSF_JWST_{filter_type}_SN_{exp_time}s.fits',

center = array((float(center_x), float(center_y)))
Sky(adu=0) # just adding zero sky offset
max_shift = array((float(10.), float(10.)))

xy_shift = array((3, 3))
PointSource(xy=Uniform(loc=center-xy_shift, scale=2*xy_shift),
            mag=Uniform(loc=mag_quasar-5, scale=10))

## these are all priors, all uniform priors
Sersic(xy=Uniform(loc=center-xy_shift, scale=2*xy_shift),
      mag=Uniform(loc=20, scale=10),
      reff=Uniform(loc=1, scale=20),
      reff_b=Uniform(loc=1, scale=20),
      index=1,
      angle=Uniform(loc=0, scale=180), angle_degrees=True)
