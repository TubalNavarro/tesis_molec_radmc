#Script que corre distintos modelos, variando parámetros.

from yso_models import *
from make_line import *
from pathlib import Path 
import shutil
import subprocess
import sys

cwd = Path.cwd().resolve()
root_dir = Path(__file__).resolve().parent

def organize_folder(modelname, molec='ch3oh'):

    if os.getcwd().endswith('%s'%(cwd)):
        pass
    else:
        os.chdir('%s'%(cwd))

    os.makedirs('%s'%(modelname), exist_ok=True)
    model_dir=Path('%s'%(modelname))
    shutil.copy(
        'inputs/dustkappa_silicate.inp',
        '%s/dustkappa_silicate.inp'%(modelname)
    )
    shutil.copy(
        'inputs/molecule_%s.inp'%molec,
        '%s/molecule_%s.inp'%(modelname, molec)
    )
    shutil.copy(
        'make_line.py',
        '%s/make_line.py'%(modelname)
    )
    shutil.copy(
        'inputs/partitionfunction_%s.inp'%molec,
        '%s/partitionfunction_%s.inp'%(modelname, molec)
    )
    shutil.copy2(
        root_dir / 'inputs' / 'G328_noise.dat',
        model_dir / 'G328_noise.dat'
    )
    
    shutil.copy2(
        root_dir / 'pv' / 'pv_G328_flipped_xy_freq.fits',
        model_dir / 'pv_G328_flipped_xy_freq.fits'
    )
    os.chdir(modelname)
    

import json

def read_pars_json(filename="parameters.json"):
    with open(filename, "r") as f:
        params = json.load(f)
    return params

def par_tests():
    params=read_pars_json("various_files/parameters.json")
    params.pop("incl")
    for incl in [0, 5, 10, 20, 30, 40, 45, 50, 60 , 70, 80, 85, 90]:
        organize_folder(f'model_Ulrich_i={incl}')
        UlrichDisk(nmodel=0, **params)      
        make_line_image_freq(incl=incl)



###### Un solo modelo, parámetros default#######
organize_folder(f'model_Ulrich_test_2')
params=read_pars_json("../various_files/test_pars.json")
incl = params.pop("incl")
UlrichDisk(nmodel=0, **params)
make_line_image_freq(62.5)