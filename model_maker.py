"""
Make all the psfmc model files with jinja
"""
print("Running model_maker.py")
from jinja2 import Environment, FileSystemLoader
import numpy as np
import sys
import os
from config_file_dict_maker import QuasarSimulationConfig
from bigfile import BigFile

sys.path.insert(1, '/fred/oz183/sberger/paper_1_bluetides/main_scripts')
from astropy.io import fits
import argparse

### static params
zp = 25.9463
###

parser = argparse.ArgumentParser(description="Which json config file should I use?")
parser.add_argument('--json_config_file', type=str, help='json config file to use')
args = parser.parse_args()

config = QuasarSimulationConfig.load_from_json(args.json_config_file)
filter_type = config.filters
include_quasar = config.include_quasar

# taking filter, quasar_mag, samp, FOV, width out of lists like they were needed in make_images_detected_quasar.py

filter_first_str = filter_type[0][-5:]
if config.include_quasar:
    mag_quasar = config.mags_AB[0]
else:
    mag_quasar = None # no mag quasar to add
ss = config.samps[0]
FOVs = config.FOVs[0]  # pkpc
widths = config.widths[0]  # arcseconds

exp_time = config.exp_time
verbose = config.verbose
noiseless_psf = config.noiseless_psf

if noiseless_psf:
    psf_name = f"NOISELESS_sci_PSF_JWST_{filter_first_str}_SN_{ss}ss_{exp_time}s.fits"
    ivm_psf_name = f'NOISELESS_ivm_PSF_JWST_{filter_first_str}_SN_{ss}ss_{exp_time}s.fits'
else:
    psf_name = f"sci_PSF_JWST_{filter_first_str}_SN_{ss}ss_{exp_time}s.fits"
    ivm_psf_name = f'ivm_PSF_JWST_{filter_first_str}_SN_{ss}ss_{exp_time}s.fits'

model_direc = f"/fred/oz183/sberger/paper_2_obs_bias/src/pipeline_full/PSFMC_MODEL_FILES/all_model_files_" + config.distinguish_string
out_file_direc = f"/fred/oz183/sberger/paper_2_obs_bias/src/pipeline_full/PSFMC_MODEL_FILES/all_out_files_" + config.distinguish_string


# making folder if it doesn't exist
if not os.path.exists(model_direc):
    os.makedirs(model_direc)
    # copy template files into model directory
    os.system("cp -r /fred/oz183/sberger/paper_2_obs_bias/src/all_template_files " + model_direc)
    print("Made model file directory.")

if not os.path.exists(out_file_direc):
    os.makedirs(out_file_direc)
    print("Made out file directory.")

indices = np.load(config.indices_to_fit)
environment = Environment(loader=FileSystemLoader(model_direc))
direc_fits = config.save_folder + "/"
direc_fits = direc_fits.replace("noiseless_", "") # images never generated with noiseless psf


### Ordering getting BH indices


def get_center_of_fits(fits_file):
    image_data_full = fits.getdata(fits_file, ext=0)
    center = np.shape(image_data_full)[0] // 2
    return center, center

i = 0
for ind in indices:
    if not include_quasar:  # just galaxy
        fits_name = f'exp_time_{exp_time}_without_quasar_ss_{ss}_not_smoothed_{ind}__with_background_JWST.NIRCAM.{filter_first_str}.fits'
        filename = model_direc + f"/without_quasar_model_{ind}.py"
        # direc_ivm_fits = f"/fred/oz183/sberger/paper_2_obs_bias/{filter_first_str}_wo_quasar_fits_files_6p5_exp_{exp_time}_FOV_{FOVs}_samp_{samps}/"
        template = environment.get_template("/all_template_files/without_quasar_model_template.py")
        ivm_fits_name = f'exp_time_{exp_time}_ivm_without_quasar_ss_{ss}_not_smoothed_{ind}__with_background_JWST.NIRCAM.{filter_first_str}.fits'
    else:  # galaxy and quasar
        fits_name = f'exp_time_{exp_time}_ss_{ss}_not_smoothed_{ind}__with_background_JWST.NIRCAM.{filter_first_str}.fits'
        filename = model_direc + f"/quasar_model_{ind}.py"
        # direc_ivm_fits = f"/fred/oz183/sberger/paper_2_obs_bias/{filter_first_str}_wo_quasar_fits_files_6p5_exp_{exp_time}_FOV_{FOVs}_samp_{samps}/"
        template = environment.get_template("/all_template_files/with_quasar_model_template.py")
        ivm_fits_name = f'exp_time_{exp_time}_ivm_ss_{ss}_not_smoothed_{ind}__with_background_JWST.NIRCAM.{filter_first_str}.fits'

    if i == 0:
        center_x, center_y = get_center_of_fits(direc_fits + fits_name)
        i = 100

    gal = {"blue_tides_num": ind,
           "filter_type": filter_type[:4],
           "zp": zp,
           "direc_fits": direc_fits,
           "direc_ivm_fits": direc_fits,
           "fits_name": fits_name,
           "psf_name": psf_name,
           "ivm_fits_name": ivm_fits_name,
           "ivm_psf_name": ivm_psf_name,
           "center_x": center_x,
           "center_y": center_y,
           "mag_quasar": mag_quasar,
           # "mag_galaxy": mag_galaxy,
           "exp_time": exp_time}

    content = template.render(gal)
    with open(filename, mode="w", encoding="utf-8") as template:
        template.write(content)
        if verbose:
            print(f"... wrote {filename}")
    template.close()
