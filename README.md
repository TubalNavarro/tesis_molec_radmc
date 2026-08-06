# molec_radmc
Molecular models in RADMC-3D


To create a single model or test single variables  collections use single_model_main.py
To get a minimization use MCMC_master.py
To get pv .fits and .png of model collections use pv_collections.py 


yso_models.py makes model and writes rdmc input files. 
make_line.py runs radmc3d and make convolved cubes. 


model_*/ directories are created after running single_model_main.py or MCMC_master.py. They contain each model input, plots and final cubes.
individual_scripts/ contains various scripts used to get intermediate products
inputs/ contains physical theoric tables for used molecules and materials. Also contains noise observational tables.
pv/ contains observational pv.fits files
  