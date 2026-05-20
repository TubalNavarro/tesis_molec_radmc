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
    


#for incl in [5,15,30, 45,60, 75]:
#    organize_folder(f'model_Hamb_i={incl}', molec='ch3oh')
#    Hamburguers()
#    make_line_image_freq(incl=incl)
#
#for r in [150,200, 300, 400, 500, 600, 1000]:
#    organize_folder(f'model_Hamb_rdisc={r}', molec='ch3oh')
#    Hamburguers(Rdisc=r)
#    make_line_image_freq()    

#for exp in [0.3,0.4,0.5,0.6,0.7]:
#    organize_folder(f'model_Hamb_dens_exp={exp}', molec='ch3oh')
#    Hamburguers(p=exp)
#    make_line_image_freq()  

organize_folder('model_Ulrich_G328_test', molec='ch3oh')
UlrichDisk(Rdisc=1000)
make_line_image_freq(incl=80)  
  

