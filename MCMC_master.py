"""
Master MCMC driver for configurable YSO models.

Select the physical model only through the project JSON:
    model.type = ulrich | hamburgers | hamburgers_piecewise
"""

import os
import psutil
import json
import uuid
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path

from config import load_config


root_dir = Path(__file__).resolve().parent



def print_memory(label=""):

    proc = psutil.Process(os.getpid())

    rss_gb = (
        proc.memory_info().rss
        / 1024**3
    )

    print(
        f"[MEM] {label} "
        f"PID={os.getpid()} "
        f"RSS={rss_gb:.3f} GiB",
        flush=True
    )

def _resolve_project_path(path_value, default_dir=None):
    path = Path(path_value)

    if path.is_absolute():
        return path

    if default_dir is not None and len(path.parts) == 1:
        return root_dir / default_dir / path

    return root_dir / path


parser = ArgumentParser()
parser.add_argument(
    "config",
    nargs="?",
    default="configs/G328.json",
    help="Project configuration JSON file",
)
args = parser.parse_args()

config_file = _resolve_project_path(args.config)
config = load_config(config_file)

model_type = config.get("model", {}).get("type", "ulrich")

os.environ["OMP_NUM_THREADS"] = str(
    config["mcmc"]["omp_threads"]
)


# Import after OMP_NUM_THREADS is set.
from yso_models import get_model_function
from make_line import make_line_image_freq
from plot_helpers import plot_walkers, plot_corner

import emcee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from astropy.io import fits


model_function = get_model_function(model_type)


# ============================================================
# EMCEE ARGUMENTS
# ============================================================

mcmc_config = config["mcmc"]
output_config = config["output"]

nthreads = mcmc_config["nthreads"]
nwalkers = mcmc_config["nwalkers"]
nburn = mcmc_config["nburn"]
nsteps = mcmc_config["nsteps"]
frac_stddev = mcmc_config["frac_stddev"]

tag_out = output_config["tag"]
test_tag = output_config["test_tag"]


# ============================================================
# MODEL FOLDERS
# ============================================================

def organize_folder(
    modelname,
    molec=config["line"]["molecule"],
):
    model_dir = (
        root_dir
        / (
            f"{tag_out}_test{test_tag}_"
            f"{nwalkers}walkers_{nsteps}steps"
        )
        / modelname
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        root_dir / "inputs" / "dustkappa_silicate.inp",
        model_dir / "dustkappa_silicate.inp",
    )

    shutil.copy2(
        root_dir / "inputs" / f"molecule_{molec}.inp",
        model_dir / f"molecule_{molec}.inp",
    )

    shutil.copy2(
        root_dir / "inputs" / f"partitionfunction_{molec}.inp",
        model_dir / f"partitionfunction_{molec}.inp",
    )

    shutil.copy2(
        root_dir / "make_line.py",
        model_dir / "make_line.py",
    )

    shutil.copy2(
        root_dir
        / "inputs"
        / config["observation"]["noise_file"],
        model_dir
        / config["observation"]["noise_file"],
    )

    shutil.copy2(
        root_dir
        / "pv"
        / config["observation"]["pv_file"],
        model_dir
        / config["observation"]["pv_file"],
    )

    return model_dir


# ============================================================
# PARAMETER SPACE
# ============================================================

parameter_file = _resolve_project_path(
    config["mcmc"]["parameter_file"],
    default_dir="configs",
)

param_space = pd.read_csv(
    parameter_file,
    skipinitialspace=True,
)
param_space.set_index("Parameter", inplace=True)

free = param_space[param_space["Fit"]]
fixed = param_space[~param_space["Fit"]]

param_names = free.index.tolist()
p0_mean = free["Mean"].to_numpy()
param_min = free["Min"].to_numpy()
param_max = free["Max"].to_numpy()
fixed_params = fixed["Mean"].to_dict()

mcmc_noise = mcmc_config["noise"]
npars = len(param_names)

p0_stddev = frac_stddev * (
    param_max - param_min
)
p0 = np.random.normal(
    p0_mean,
    p0_stddev,
    size=(nwalkers, npars),
)


# ============================================================
# SAMPLE ANALYSIS
# ============================================================

def error_samples(samples, best_params, perc=34.1):
    errpos, errneg = [], []

    for i in range(npars):
        tmp = best_params[i]

        indpos = samples[:, i] > tmp
        indneg = samples[:, i] < tmp

        val = samples[:, i][indpos] - tmp
        errpos.append(
            np.percentile(val, [2 * perc])
        )

        val = np.abs(
            samples[:, i][indneg] - tmp
        )
        errneg.append(
            np.percentile(val, [2 * perc])
        )

    return errneg, errpos


def analyse_samples(
    sampler,
    nstats=None,
    make_walkers=True,
    make_corner=True,
):
    if nstats is None:
        samples = sampler.chain
    else:
        samples = sampler.chain[:, -nstats:]

    samples = samples.reshape(
        -1,
        samples.shape[-1],
    )

    best_params = np.median(
        samples,
        axis=0,
    )

    errneg, errpos = error_samples(
        samples,
        best_params,
    )

    np.savetxt(
        (
            f"test{test_tag}_log_pars_"
            f"{tag_out}_cube_"
            f"{nwalkers}walkers_{nsteps}steps.txt"
        ),
        np.array(
            [p0_mean, best_params, errneg, errpos],
            dtype="object",
        ),
        fmt="%.8f",
        header=str(param_names),
    )

    np.savetxt(
        (
            f"test{test_tag}_parameter_samples_"
            f"{tag_out}_cube_"
            f"{nwalkers}walkers_{nsteps}steps.txt"
        ),
        samples,
        fmt="%.8f",
        header=str(param_names),
    )

    if make_walkers:
        plot_walkers(
            sampler.chain.T,
            best_params,
            header=param_names,
        )
        plt.tight_layout()
        plt.savefig(
            (
                f"test{test_tag}_mc_walkers_"
                f"{tag_out}_"
                f"{nwalkers}walkers_{nsteps}steps.png"
            ),
            dpi=300,
        )
        plt.close()

    if make_corner:
        plot_corner(
            samples,
            labels=param_names,
        )
        plt.savefig(
            (
                f"test{test_tag}_mc_corner_"
                f"{tag_out}_"
                f"{nwalkers}walkers_{nsteps}steps.png"
            ),
            dpi=300,
        )
        plt.close()


# ============================================================
# CLEANUP
# ============================================================

def clean_model_folder(model_dir):
    for file in model_dir.glob("*.inp"):
        file.unlink()

    for pattern in [
        "raw_radmc*",
        "radmc_*",
        "_convolved*.fits",
        "csub_convolved_J*.fits",
        "cont*.fits",
    ]:
        for file in model_dir.glob(pattern):
            file.unlink()

    files_to_delete = [
        "image_line.out",
        "dust_temperature.dat",
        "G328.dat",
        "radmc3d.out",
    ]

    for filename in files_to_delete:
        file = model_dir / filename
        if file.exists():
            file.unlink()


# ============================================================
# OBSERVATIONAL PV
# ============================================================

real_file = (
    root_dir
    / "pv"
    / config["observation"]["pv_file"]
)

with fits.open(real_file) as hdu_obs:
    data = hdu_obs[0].data.copy()


# ============================================================
# LIKELIHOOD
# ============================================================

def ln_likelihood(params):
    print_memory("START likelihood")
    if (
        np.any(params <= param_min)
        or np.any(params >= param_max)
    ):
        return -np.inf

    param_dict = fixed_params.copy()
    param_dict.update(
        zip(param_names, params)
    )

    pid = os.getpid()
    model_id = uuid.uuid4().hex[:10]
    modelname = (
        f"{model_type}_{pid}_{model_id}"
    )

    model_dir = organize_folder(
        modelname,
        molec=config["line"]["molecule"],
    )

    with open(
        model_dir / "parameters.json",
        "w",
    ) as f:
        json.dump(
            {
                k: float(v)
                for k, v in param_dict.items()
            },
            f,
            indent=4,
        )

    try:
        os.chdir(model_dir)

        inclination = param_dict["incl"]
        
        model_function(
            nmodel=model_id,
            lum_bol=config["source"]["lum_bol"],
            molec=config["line"]["molecule"],
            grid_config=config["physical_grid"],
            model_config=config["model_options"],
            radmc_config=config["radmc3d"],
            **{
                k: v
                for k, v in param_dict.items()
                if k != "incl"
            },
        )
        
        print_memory("AFTER UlrichDisk")
        
        pv_model = make_line_image_freq(
            incl=inclination,
            config=config,
        )
        print_memory("AFTER make_line")

    except Exception as error:
        with open(
            model_dir / "ERROR.txt",
            "w",
        ) as f:
            f.write(
                f"{type(error).__name__}: {error}\n"
            )

        return -np.inf

    finally:
        os.chdir(root_dir)

    lnx2 = -0.5 * np.sum(
        ((data - pv_model) / mcmc_noise) ** 2
    )

    with open(
        model_dir / "likelihood.txt",
        "w",
    ) as f:
        f.write(
            f"ln_likelihood = {lnx2}\n"
        )

    clean_model_folder(model_dir)

    print_memory("END likelihood")
    
    return (
        lnx2
        if np.isfinite(lnx2)
        else -np.inf
    )


# ============================================================
# RUN SAMPLER
# ============================================================

def run_mcmc(
    p0,
    nwalkers,
    nburn,
    nsteps,
    npars,
    nthreads=None,
):
    with Pool(
        processes=nthreads
    ) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            npars,
            ln_likelihood,
            pool=pool,
        )

        # Burn-in remains disabled exactly as in the current script.
        # state = sampler.run_mcmc(
        #     p0,
        #     nburn,
        #     progress=True,
        # )
        # sampler.reset()

        print(
            f"\nRunning {model_type} production: "
            f"{nsteps} steps"
        )

        start = time.time()

        sampler.run_mcmc(
            p0,
            nsteps,
            progress=True,
        )

        print(
            "Production took "
            f"{time.time() - start:.1f} seconds"
        )

    return sampler


if __name__ == "__main__":
    print(f"Configuration: {config_file}")
    print(f"Model type: {model_type}")
    print(f"Parameter file: {parameter_file}")

    sampler = run_mcmc(
        p0,
        nwalkers,
        nburn,
        nsteps,
        npars,
        nthreads=nthreads,
    )

    analyse_samples(sampler)
