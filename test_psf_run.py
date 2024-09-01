from psfMC import model_galaxy_mcmc
import os
import glob
from mpi4py import MPI
from multiprocessing import Pool
import numpy as np
import sys
from config_file_dict_maker import QuasarSimulationConfig
import argparse


"""Script to use psfmc on many galaxies using MPI"""

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# print(" Running %d parallel MPI processes" % size)
#
# comm.Barrier()                    ### Start stopwatch ###
# t_start = MPI.Wtime()

# These are additional parameters for the MCMC fitter. Number of iterations,
# number of burn-in iterations (which are discarded)
_mcparams = {'burn': 1000, 'iterations': 2000, 'chains': 50}
# _mcparams = {'burn': 200, 'iterations': 400, 'chains': 50}

def run_mcmc(model_file):
    output_name = model_file.replace('model', 'out').replace('.py', '')
    if len(glob.glob(output_name +'_*fits*')) > 0:
        print(glob.glob(output_name +'*fits*'))
        print('{} already processed, skipping'.format(model_file))
        return
    model_galaxy_mcmc(model_file, output_name=output_name, **_mcparams)

# only if using all galaxies
# luminous_BH_indices = np.load(
#         "/fred/oz183/sberger/paper_1_bluetides/param_npy_files/luminous_BH_indices.npy")  # indexes of BH with largest accretion rate
# num_galaxies_per_task = luminous_BH_indices / size
# model_files = glob.glob("/fred/oz183/sberger/paper_2_obs_bias/all_model_files/without_quasar_model_*.py")

# print("Processing BH #...")
# print(ind_current)
# print("on rank # ", comm.rank)
model_file = f'/fred/oz183/sberger/paper_2_obs_bias/src/test_psf.py'
print("Running model file {}".format(model_file))
try:
    run_mcmc(model_file)
except:
    pass
