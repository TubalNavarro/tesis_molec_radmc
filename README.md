# molec_radmc
Molecular models in RADMC-3D


To create a single model or test single variables  collections use single_model_main.py
To get a minimization use MCMC_master.py
To get pv .fits and .png of model collections use pv_collections.py 


yso_models.py makes model and writes rdmc input files. 
make_line.py runs radmc3d and make convolved cubes. 


model_*/ directories are created after running single_model_main.py or MCMC_master.py (under mcmc_*). They contain each model input, plots and final cubes.
individual_scripts/ contains various scripts used to get intermediate products
inputs/ contains physical theoric tables for used molecules and materials. Also contains noise observational tables.
pv/ contains observational pv.fits files

## Automatization update 24/aug/2026

Now you create models trough a .JSON parameter file, under configs/ directory.
To create a model different from the default, change value of config_file="" in single_model_main.py. In MCMC_master.py change the default in parser argument "config" or use argument in terminal.

### Parameter file structure

* Source: Name and physical properties of the observed source. 
* line: Line properties such as molecule name (must be the same as used in the input files), transtion number (must be as in the input files), rest frecuency of the observational cube, and line rest frecuency. 
* observation: pv and noise files constructed from the observed cube. Must be put at pv/ and inputs/ folders, respectively.
* physical grid: Size and number of points of the simulation. It is used the same physical resolution (cell size) as the one used in the observational pv file. The number of points must be 4 more than the spacial Npix in the pv file.
* model options: Ulrich + Disk options; true or false disc and envelope, gtd ratio, microturbulence an rho_min. 
* radmc3d: Radiative transfer options. Normally don't modify these.
* synthetic_image: radmc3d image construction options. npix is the same as npoints in grid. nchan and dv_kms depends on the spectral resolution of observed pv. pixel_scale_arcsec is the same as spatial arcsec resolution of the pv.
* beam: observational beam  properties.
* pv: Properties of the synthetic pv to construct. Spacing and width must be as in the observational pv. pa_ is 90 default. length_arcsec must be spacing*npix_pv_observed+ 01 decimal digits (if not the pv has -1 pixels)  
* mcmc: minimization parameters. Careful to not use more threads than needed. If nthreads>nwalkers it can let zombie processes to consume all RAM.  Default (secure) is nthreads=nwalkers/2. nburn can be used to do a burning stage and then reset the mcmc run. Now disabled.(commented in MCMC_master). must give parameter csv file for the run. For now n_cpu's is aprox ntrheads/2, so for models with 130 pix, 48 walkers and 24 workers, 12 cpu's are full and a total of 45 Gib Mem. 
* output: tags for the run.

### Paralelization parameters
* nthreads: 

### to do
* Script that construct JSON file from observational cube or pv header