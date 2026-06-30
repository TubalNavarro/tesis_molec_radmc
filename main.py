#Script que corre distintos modelos, variando parámetros.

from yso_models import *
from make_line import *
from pathlib import Path 
import shutil
import subprocess
import sys

cwd = Path.cwd().resolve()
from Line_max import analyze_fits_cube


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
    



for incl in [0, 5, 10, 20, 30, 40, 45, 50, 60 , 70, 80, 85, 90]:
    organize_folder(f'model_Ulrich_nodisk_i={incl}', molec='ch3oh')
    UlrichDisk(discFlag=False)
    make_line_image_freq(incl=incl)

for r in [200, 300, 400, 500, 600, 1000, 1500]:
    organize_folder(f'model_Ulrich_nodisk_rdisc={r}', molec='ch3oh')
    UlrichDisk(Rdisc=r, discFlag=False)
    make_line_image_freq()    
for A in [1,3,5,7, 9]:
    organize_folder(f'model_Ulrich_dens_nodisk_Arho0={A}', molec='ch3oh')
    UlrichDisk(Arho0=A, discFlag=False)
    make_line_image_freq()  


for M in [9,15,18,20,22,24,27]:
    organize_folder(f'model_Ulrich_nodisk_MStar={M}', molec='ch3oh')
    UlrichDisk(MStar=M, discFlag=False)
    make_line_image_freq() 

for Mdot in [5e-5,8e-5,1e-4,3e-4,6e-4, 8e-4 ,2e-3]:
    organize_folder(f'model_Ulrich_nodisk_Mrate={Mdot}', molec='ch3oh')
    UlrichDisk(MRate=Mdot, discFlag=False)
    make_line_image_freq() 

for B in [0.5, 1, 3, 5, 7, 10]:
    organize_folder(f'model_Ulrich_nodisk_BT={B}', molec='ch3oh')
    UlrichDisk(BT=B, discFlag=False)
    make_line_image_freq() 

#for T10 in [2000, 2500, 3000, 3200, 3500, 4000]:
#    organize_folder(f'model_Ulrich_nodisk_T10={T10}', molec='ch3oh')
#    UlrichDisk(T10Env=T10, discFlag=False)
#    make_line_image_freq() 

for exp in [1,1.25,1.5,1.75,2,2.25,2.5,2.75,3, 3.25,3.5]:
    organize_folder(f'model_Ulrich_nodisk_exp={exp}', molec='ch3oh')
    UlrichDisk(exp_disc=exp, discFlag=False)
    make_line_image_freq()
for X in [1e-6,2.2e-6, 5e-6, 1e-5, 2.2e-5]:
    organize_folder(f'model_Ulrich_nodisk_exp={X}', molec='ch3oh')
    UlrichDisk(molec_abund=X   , discFlag=False)
    make_line_image_freq()
for incl in [0, 5, 10, 20, 30, 40, 45, 50, 60 , 70, 80, 85, 90]:
    organize_folder(f'model_Ulrich_i={incl}', molec='ch3oh')
    UlrichDisk()
    make_line_image_freq(incl=incl)

for r in [200, 300, 400, 500, 600, 1000, 1500]:
    organize_folder(f'model_Ulrich_rdisc={r}', molec='ch3oh')
    UlrichDisk(Rdisc=r)
    make_line_image_freq()    
for A in [1,3,5,7, 9]:
    organize_folder(f'model_Ulrich_dens_Arho0={A}', molec='ch3oh')
    UlrichDisk(Arho0=A)
    make_line_image_freq() 
     
#for T10 in [2000, 2500, 3000, 3200, 3500, 4000]:
#    organize_folder(f'model_Ulrich_T10={T10}', molec='ch3oh')
#    UlrichDisk(T10Env=T10)
#    make_line_image_freq() 

for M in [9,15,18,20,22,24,27]:
    organize_folder(f'model_Ulrich_MStar={M}', molec='ch3oh')
    UlrichDisk(MStar=M)
    make_line_image_freq() 

for Mdot in [5e-5,8e-5,1e-4,3e-4,6e-4, 8e-4 ,2e-3]:
    organize_folder(f'model_Ulrich_Mrate={Mdot}', molec='ch3oh')
    UlrichDisk(MRate=Mdot)
    make_line_image_freq() 

for B in [0.5, 1, 3, 5, 7, 10]:
    organize_folder(f'model_Ulrich_BT={B}', molec='ch3oh')
    UlrichDisk(BT=B)
    make_line_image_freq() 
for exp in [1,1.25,1.5,1.75,2,2.25,2.5,2.75,3, 3.25,3.5]:
    organize_folder(f'model_Ulrich_exp={exp}', molec='ch3oh')
    UlrichDisk(exp_disc=exp)
    make_line_image_freq()