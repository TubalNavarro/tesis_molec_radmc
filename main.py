#Script that creates multiple models in separate folders

from yso_models import *
from make_line import *
from pathlib import Path 
import shutil
import subprocess
import sys

cwd = Path.cwd().resolve()
from Line_max import analyze_fits_cube
#Function to create a folder for a given model and copy dust and line (.inp) properties
def organize_folder(modelname, molec='co'):

    if os.getcwd().endswith('%s'%(cwd)):
        pass
    else:
        os.chdir('%s'%(cwd))

    os.makedirs('%s'%(modelname), exist_ok=True)
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
    os.chdir(modelname)

organize_folder('model_Ulrich', molec='ch3oh')
UlrichDisk(molec='ch3oh')

make_line_image_freq()






