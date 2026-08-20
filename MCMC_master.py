'''
Master script for RL modelling of G328
Ulrich+disk model that uses ulrichdisk_cont and ulrichdisk_pv
'''
import os
import uuid
os.environ["OMP_NUM_THREADS"] = "2"


#***************
#Main libraries
#***************
from yso_models import *
from make_line import *
from plot_helpers import plot_walkers, plot_corner
#***********************
#Libs for parameter seed
#***********************
import pandas as pd
from sklearn.model_selection import ParameterGrid
#*************************
#Emcee and multiprocessing
#*************************
import emcee
import multiprocessing
from multiprocessing import Pool, Process
#*******************
#Addtional libraries
#*******************
from argparse import ArgumentParser
from astropy.io import fits
import numpy as np
import time
import sys
import json

from pathlib import Path
import shutil

root_dir = Path(__file__).resolve().parent

######function that creates a new folder for each model and imports necessary inputs

#***************
#EMCEE ARGUMENTS
#***************
nthreads = 16 #if None use max num of threads
nwalkers = 32 #Number of different models invoked by emcee, these will evolve over nsteps 
nburn=1
nsteps = 100
tag_out = 'mcmc_G328'
frac_stddev = 1e-2
n_test=14
#frac_stddev is the fraction of parameter range to calculate stddev of the initial seed of parameters,
# e.g. sigma0_parA = frac_stddev*(bound1_parA-bound0_parA)


def organize_folder(modelname, molec='ch3oh'):

    model_dir = root_dir / f'mcmc_models_test{n_test}' / modelname

    model_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    shutil.copy2(
        root_dir / 'inputs' / 'dustkappa_silicate.inp',
        model_dir / 'dustkappa_silicate.inp'
    )

    shutil.copy2(
        root_dir / 'inputs' / f'molecule_{molec}.inp',
        model_dir / f'molecule_{molec}.inp'
    )

    shutil.copy2(
        root_dir / 'inputs' / f'partitionfunction_{molec}.inp',
        model_dir / f'partitionfunction_{molec}.inp'
    )

    shutil.copy2(
        root_dir / 'make_line.py',
        model_dir / 'make_line.py'
    )

    shutil.copy2(
        root_dir / 'inputs' / 'G328_noise.dat',
        model_dir / 'G328_noise.dat'
    )

    shutil.copy2(
        root_dir / 'pv' / 'pv_G328_flipped_xy_freq.fits',
        model_dir / 'pv_G328_flipped_xy_freq.fits'
    )

    return model_dir


#*******************
#INIT PARS FOR EMCEE
#*******************
# Read CSV file with parameter space

param_space = pd.read_csv(root_dir / 'free_params_ulrich.csv', skipinitialspace=True)
param_space.set_index('Parameter', inplace=True)

free = param_space[param_space['Fit']]
fixed = param_space[~param_space['Fit']]

param_names = free.index.tolist()
p0_mean = free['Mean'].to_numpy()
param_min = free['Min'].to_numpy()
param_max = free['Max'].to_numpy()
fixed_params = fixed['Mean'].to_dict()

noise = 5e-3 #
npars = len(param_names)
p0_stddev = frac_stddev * (param_max - param_min)
p0 = np.random.normal(p0_mean, p0_stddev, size=(nwalkers, npars))

def error_samples(samples, best_params, perc=34.1):
    errpos, errneg = [], []
    for i in range(npars):
        tmp = best_params[i]
        indpos = samples[:,i] > tmp
        indneg = samples[:,i] < tmp
        val = samples[:,i][indpos] - tmp
        errpos.append(np.percentile(val, [2*perc])) #1 sigma (2x perc 34.1), positive pars 
        val = np.abs(samples[:,i][indneg] - tmp)
        errneg.append(np.percentile(val, [2*perc]))
    return errneg, errpos
        
def analyse_samples(sampler, nstats=None, make_walkers=True, make_corner=True):
    #sampler.chain shape: nwalkers, nsteps, npars
    import matplotlib.pyplot as plt
    if nstats is None: samples = sampler.chain 
    else: samples = sampler.chain[:,-nstats:]
    samples = samples.reshape(-1, samples.shape[-1]) #reshape samples to nwalkers*nsteps, npars
    best_params = np.median(samples, axis=0)
    errneg, errpos = error_samples(samples, best_params)
    np.savetxt(f'test{n_test}_log_pars_{tag_out}_cube_{nwalkers}walkers_{nsteps}steps.txt',
               np.array([p0_mean, best_params, errneg, errpos], dtype='object'),
               fmt='%.8f', header=str(param_names))
    np.savetxt(f'test{n_test}_parameter_samples_{tag_out}_cube_{nwalkers}walkers_{nsteps}steps.txt', 
               samples,
               fmt='%.8f', header=str(param_names))

    if make_walkers: 
        plot_walkers(sampler.chain.T, best_params, header=param_names)
        plt.tight_layout()
        plt.savefig(f'test{n_test}_mc_walkers_{tag_out}_{nwalkers}walkers_{nsteps}steps.png', dpi=300)
        plt.close()
    if make_corner:
        plot_corner(samples, labels=param_names)
        plt.savefig(f'test{n_test}_mc_corner_{tag_out}_{nwalkers}walkers_{nsteps}steps.png', dpi=300)
        plt.close()

def clean_model_folder(model_dir):
    # Borrar todos los .inp
    for file in model_dir.glob('*.inp'):
        file.unlink()
    for file in model_dir.glob('raw_radmc*'):
        file.unlink()
    for file in model_dir.glob('radmc_*'):
        file.unlink()
    for file in model_dir.glob('_convolved*.fits'):
        file.unlink()
    for file in model_dir.glob('csub_convolved_J*.fits'):
        file.unlink()
    for file in model_dir.glob('cont*.fits'):
        file.unlink()
    # Borrar archivos específicos
    files_to_delete = [
        'image_line.out',
        'dust_temperature.dat',
        'G328.dat',
        'radmc3d.out'
    ]

    for filename in files_to_delete:
        file = model_dir / filename
        if file.exists():
            file.unlink()


#***********************
#READ DATA AND SET PATHS
#***********************

# Leer imagen real tomada con telecopio ALMA
real_file = root_dir / 'pv' / 'pv_G328_flipped_xy_freq.fits'

with fits.open(real_file) as hdu_obs:
    data = hdu_obs[0].data.copy()

#*****************
#RUN EMCEE SAMPLER
#*****************
def ln_likelihood(params):
    if np.any(params <= param_min) or np.any(params >= param_max):
        return -np.inf
    param_dict = fixed_params.copy()
    param_dict.update(zip(param_names, params))
    pid = os.getpid()
    model_id = uuid.uuid4().hex[:10]
    modelname = f'model_{pid}_{model_id}'
    model_dir = organize_folder(modelname, molec='ch3oh')
    with open(model_dir / 'parameters.json', 'w') as f:
        json.dump({k: float(v) for k, v in param_dict.items()}, f, indent=4)
   
    try:
        os.chdir(model_dir)

        inclination = param_dict['incl']

        UlrichDisk(
            nmodel=model_id,
            **{k: v for k, v in param_dict.items() if k != 'incl'}
        )

        pv_model = make_line_image_freq(incl=inclination)

    except Exception as error:
        with open(model_dir / 'ERROR.txt', 'w') as f:
            f.write(f'{type(error).__name__}: {error}\n')
        return -np.inf

    finally:
        os.chdir(root_dir)

    lnx2 = -0.5 * np.sum(((data - pv_model) / noise)**2)
    
    with open(model_dir / 'likelihood.txt', 'w') as f:
        f.write(f'ln_likelihood = {lnx2}\n')
    
    clean_model_folder(model_dir)
    
    return lnx2 if np.isfinite(lnx2) else -np.inf


def run_mcmc(p0, nwalkers, nburn, nsteps, npars, nthreads=None):
    with Pool(processes=nthreads) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, npars, ln_likelihood, pool=pool
        )

        #print(f"\nRunning burn-in: {nburn} steps")
        #start = time.time()
        #state = sampler.run_mcmc(p0, nburn, progress=True)
        #print(f"Burn-in took {time.time() - start:.1f} seconds")

        #sampler.reset()

        print(f"\nRunning production: {nsteps} steps")
        start = time.time()
        sampler.run_mcmc(p0, nsteps, progress=True)
        print(f"Production took {time.time() - start:.1f} seconds")

    return sampler

if __name__ == '__main__':
    sampler = run_mcmc(p0, nwalkers,nburn, nsteps, npars, nthreads=nthreads)
    analyse_samples(sampler)