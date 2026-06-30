# molec_radmc
Molecular models in RADMC-3D

yso_models.py makes model and writes rdmc input files. 
make_line.py runs radmc3d and make convolved cubes. 
main.py controls yso_models and make_line. 

model_* directories contains each model inputs, plots and final cubes


# Update June 30

main_parallel.py runs the model tests in parallel

To get the .png pv diagrams and residuals, in a separate folder, run pv_collections.py