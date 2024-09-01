"""
This script images specified BlueTides galaxies with our without quasar added as described in the JSON config file
used.
"""
print("Running get_all_fits_wrapper.py")

import sys
import pickle
sys.path.insert(1, '/fred/oz183/sberger/paper_1_bluetides/main_scripts')
from make_images_detected_quasar import MakeImages
from config_file_dict_maker import QuasarSimulationConfig
from dataholder import BlueTidesDataHolder
import argparse
from get_gal_lums import GetGalLums
import numpy as np

parser = argparse.ArgumentParser(description="Which json config file should I use?")
parser.add_argument('--json_config_file', type=str, help='json config file to use')
args = parser.parse_args()

config = QuasarSimulationConfig.load_from_json(args.json_config_file)

indices_to_image = np.load(config.indices_to_fit)

if config.redshift != 6.5:
    print("not yet set up")
    exit()
else:
    print(f"redshift is {config.redshift}")
    dataholder_file_name = f"/fred/oz183/sberger/paper_1_bluetides/main_scripts/bluetides_glory/bluetides_6p5.pkl"
    with open(dataholder_file_name, 'rb') as f:
        dataholder_curr = pickle.load(f)
    print(f"Loaded {dataholder_file_name} for z = {config.redshift}. Starting processing...")

IMAGER = MakeImages(dataholder=dataholder_curr,
        filters=config.filters,
        mags_AB=config.mags_AB,
        samps=config.samps,
        FOVs=config.FOVs,
        include_quasar=config.include_quasar,
        ding=config.ding,
        exp_time=config.exp_time,
        save_folder=config.save_folder,
        create_background=config.create_background,
        choose_samp_4_me=config.choose_samp_4_me,
        smooth=config.smooth,
        dust=config.dust,
        widths=config.widths,
        use_MPI=config.use_MPI,
        verbose=config.verbose,
        rank=config.rank,
        size=config.size,
        comm=config.comm,
        just_quasar=config.just_quasar,
        shot=config.shot,
        indices_to_image=indices_to_image
    )


IMAGER.populate_data_for_galaxies_and_image(save_intermediate_data=False)