"""
This script makes the JSON configuration file to be used in model_maker.py to generate all model files for psfmc.
"""
print("Running config_file_dict_maker.py")

import json
import argparse
import os
import subprocess

class QuasarSimulationConfig:
    def __init__(self, run_through_psfmc, redshift, filters, mags_AB, samps, FOVs, include_quasar, exp_time,
                create_background=True, choose_samp_4_me=False,
                 smooth=False, dust=True, widths=None, use_MPI=True, verbose=False, rank=None,
                 size=None, comm=None, shot=True, length_of_bhar_using=108000, ding=False,
                 distinguish_string=None, save_folder=None, just_quasar=False, noiseless_psf=False,
                 generate_images=False, indices_to_fit=[0]):

        ## MUST BE LISTS
        self.filters = filters
        self.mags_AB = mags_AB
        self.samps = samps
        self.FOVs = FOVs
        self.widths = widths

        self.run_through_psfmc = run_through_psfmc
        self.redshift = redshift
        self.include_quasar = include_quasar

        self.exp_time = exp_time
        self.create_background = create_background
        self.choose_samp_4_me = choose_samp_4_me
        self.smooth = smooth
        self.dust = dust
        self.widths = widths
        self.use_MPI = use_MPI
        self.verbose = verbose
        self.rank = rank
        self.size = size
        self.comm = comm
        self.shot = shot
        self.length_of_bhar_using = length_of_bhar_using
        self.ding = ding
        self.distinguish_string = distinguish_string # None until defined below
        self.save_folder = save_folder # None until defined below
        self.just_quasar = just_quasar
        self.noiseless_psf = noiseless_psf
        self.generate_images = generate_images
        self.indices_to_fit = indices_to_fit


        if self.include_quasar:
            mag_str = str(self.mags_AB[0]).replace(".", "p")
        filter_first_str = self.filters[0][-4:]
        exp_time_str = str(exp_time)
        fovs_str = str(self.FOVs[0])
        samps_str = str(self.samps[0])
        if self.distinguish_string == None: # only run on first pass
            if self.include_quasar: # galaxy + quasar
                    if self.noiseless_psf:
                        self.distinguish_string = f"noiseless_{filter_first_str}_{mag_str}_full_quasar_fits_files_6p5_exp_{exp_time_str}_FOV_{fovs_str}_samp_{samps_str}"
                    else:
                        self.distinguish_string = f"{filter_first_str}_{mag_str}_full_quasar_fits_files_6p5_exp_{exp_time_str}_FOV_{fovs_str}_samp_{samps_str}"
                    self.save_folder = f"/fred/oz183/sberger/paper_2_obs_bias/fits_files_w_quasar/" + self.distinguish_string
            else: # just galaxy
                    self.distinguish_string = f"{filter_first_str}_wo_quasar_fits_files_6p5_exp_{exp_time_str}_FOV_{fovs_str}_samp_{samps_str}"
                    self.save_folder = f"/fred/oz183/sberger/paper_2_obs_bias/fit_files_gal_only/" + self.distinguish_string

    def to_dict(self):
        return self.__dict__

    def save_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def load_from_json(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)