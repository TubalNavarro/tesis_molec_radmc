"""
Run individual Ulrich models or vary one parameter at a time.
"""

import os
import json
import shutil
from pathlib import Path

from yso_models import UlrichDisk
from make_line import make_line_image_freq
from config import load_config



root_dir = Path(__file__).resolve().parent

config_file = root_dir / "configs" / "G328.json"
config = load_config(config_file)



def read_pars_json(filename):

    filename = Path(filename)

    if not filename.is_absolute():
        filename = root_dir / filename

    with open(filename, "r") as f:
        params = json.load(f)

    return params



def organize_folder(modelname, config):

    molec = config["line"]["molecule"]

    model_dir = root_dir / modelname

    model_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    shutil.copy2(
        root_dir / "inputs" / "dustkappa_silicate.inp",
        model_dir / "dustkappa_silicate.inp"
    )

    shutil.copy2(
        root_dir / "inputs" / f"molecule_{molec}.inp",
        model_dir / f"molecule_{molec}.inp"
    )

    shutil.copy2(
        root_dir / "inputs" / f"partitionfunction_{molec}.inp",
        model_dir / f"partitionfunction_{molec}.inp"
    )

    noise_file = (
        root_dir / "inputs" / config["observation"]["noise_file"]
    )

    shutil.copy2(
        noise_file,
        model_dir / noise_file.name
    )

    pv_file = (
        root_dir/ "pv" / config["observation"]["pv_file"]
    )

    shutil.copy2(
        pv_file,
        model_dir / pv_file.name
    )

    return model_dir


def run_model(modelname, params, config):

    params = params.copy()

    incl = params.pop("incl")

    model_dir = organize_folder(
        modelname=modelname,
        config=config
    )

    try:

        os.chdir(model_dir)

        print("\n===================================")
        print(f"Running model: {modelname}")
        print(f"Inclination: {incl}")
        print("===================================\n")

        UlrichDisk(
            nmodel=modelname,

            molec=config["line"]["molecule"],
            lum_bol=config["source"]["lum_bol"],
            grid_config=config["physical_grid"],
            model_config=config["model_options"],
            radmc_config=config["radmc3d"],

            **params
        )

        pv_model = make_line_image_freq(
            incl=incl,
            config=config
        )

    finally:

        os.chdir(root_dir)

    return pv_model



def single_model(
    params_file="configs/test_pars_G328.json",
    modelname=f"model_Ulrich_test_noenv2"
):

    params = read_pars_json(
        params_file
    )

    return run_model(
        modelname=modelname,
        params=params,
        config=config
    )



def parameter_test(
    parameter,
    values,
    params_file="configs/test_pars_G328.json"
):

    base_params = read_pars_json(
        params_file
    )

    for value in values:

        params = base_params.copy()

        params[parameter] = value

        modelname = (
            f"model_Ulrich_"
            f"{parameter}={value:g}"
        )

        run_model(
            modelname=modelname,
            params=params,
            config=config
        )


if __name__ == "__main__":

    single_model()
   
    #parameter_test( "Rdisc", [150,200, 300, 500,600, 700,  800 ])
    #parameter_test( "incl", [5, 20, 30, 45, 60, 70, 90])
    #parameter_test("p",[0.3,0.6, 1])
    #parameter_test("MRate",[5e-4, 8e-4, 1e-3, 2e-3])