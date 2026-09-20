"""
Run an individual configured YSO model or vary one parameter at a time.

The model family is selected only from the project JSON:
    model.type = ulrich | hamburgers | hamburgers_piecewise
"""

import os
import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

from yso_models import get_model_function
from make_line import make_line_image_freq
from config import load_config


root_dir = Path(__file__).resolve().parent


def _resolve_project_path(path_value, default_dir=None):
    path = Path(path_value)

    if path.is_absolute():
        return path

    if default_dir is not None and len(path.parts) == 1:
        return root_dir / default_dir / path

    return root_dir / path


def read_pars_json(filename):
    filename = _resolve_project_path(
        filename,
        default_dir="configs",
    )

    with open(filename, "r") as f:
        params = json.load(f)

    return params


def get_model_type(config):
    return config.get("model", {}).get("type", "ulrich")


def organize_folder(modelname, config):
    molec = config["line"]["molecule"]

    model_dir = root_dir / modelname

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

    noise_file = (
        root_dir
        / "inputs"
        / config["observation"]["noise_file"]
    )

    shutil.copy2(
        noise_file,
        model_dir / noise_file.name,
    )

    pv_file = (
        root_dir
        / "pv"
        / config["observation"]["pv_file"]
    )

    shutil.copy2(
        pv_file,
        model_dir / pv_file.name,
    )

    return model_dir


def run_model(modelname, params, config):
    params = params.copy()

    incl = params.pop("incl")

    model_type = get_model_type(config)
    model_function = get_model_function(model_type)

    model_dir = organize_folder(
        modelname=modelname,
        config=config,
    )

    try:
        os.chdir(model_dir)

        print("\n===================================")
        print(f"Running model: {modelname}")
        print(f"Model type: {model_type}")
        print(f"Inclination: {incl}")
        print("===================================\n")

        model_function(
            nmodel=modelname,
            molec=config["line"]["molecule"],
            lum_bol=config["source"]["lum_bol"],
            grid_config=config["physical_grid"],
            model_config=config["model_options"],
            radmc_config=config["radmc3d"],
            **params,
        )

        pv_model = make_line_image_freq(
            incl=incl,
            config=config,
        )

    finally:
        os.chdir(root_dir)

    return pv_model


def single_model(
    config,
    params_file=None,
    modelname=None,
):
    model_type = get_model_type(config)
    single_cfg = config.get("single_model", {})

    if params_file is None:
        params_file = single_cfg.get("parameter_file")

    # Backward compatibility with the old Ulrich-only layout.
    if params_file is None and model_type == "ulrich":
        params_file = "test_pars_G328.json"

    if params_file is None:
        raise ValueError(
            "No single-model parameter file was supplied. "
            "Add single_model.parameter_file to the project JSON "
            "or pass --params."
        )

    if modelname is None:
        modelname = single_cfg.get(
            "modelname",
            f"model_{model_type}_test",
        )

    params = read_pars_json(params_file)

    return run_model(
        modelname=modelname,
        params=params,
        config=config,
    )


def parameter_test(
    config,
    parameter,
    values,
    params_file=None,
):
    model_type = get_model_type(config)
    single_cfg = config.get("single_model", {})

    if params_file is None:
        params_file = single_cfg.get("parameter_file")

    if params_file is None and model_type == "ulrich":
        params_file = "test_pars_G328.json"

    if params_file is None:
        raise ValueError(
            "parameter_test needs a parameter JSON. "
            "Add single_model.parameter_file to the project JSON "
            "or pass params_file."
        )

    base_params = read_pars_json(params_file)

    for value in values:
        params = base_params.copy()
        params[parameter] = value

        modelname = (
            f"model_{model_type}_"
            f"{parameter}={value:g}"
        )

        run_model(
            modelname=modelname,
            params=params,
            config=config,
        )


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "config",
        nargs="?",
        default="configs/G328_ulrich.json",
        help="Project configuration JSON file",
    )

    parser.add_argument(
        "--params",
        default=None,
        help=(
            "Optional single-model parameter JSON. "
            "Overrides single_model.parameter_file."
        ),
    )

    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Optional model output directory name. "
            "Overrides single_model.modelname."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config_file = _resolve_project_path(args.config)
    config = load_config(config_file)

    single_model( config=config, params_file=args.params, modelname=args.name)
    
    #parameter_test(config, "incl", [5, 20, 30, 45, 60, 70, 90])
    #parameter_test(config, "cavity_ang",[0,10,20,30,50,60,80,85])
    #parameter_test(config, "Rdisc", [150, 540, 780, 2000, 3000])
    #parameter_test(config, "MStar",[10,15,20,25,30,60])    
    #parameter_test(config, "molec_abund",[1e-6, 2e-6, 5e-6, 1e-5, 2e-5])
    #parameter_test(config, "MRate",[10,15,20,25,30,60])
    #parameter_test(config, "p",[0.4,0.5,0.6,0.66,0.8]) 
    #parameter_test(config, "T10Env",[3600,3000,2500,2000,1500,1000,700])  