# molec_radmc
Molecular models in RADMC-3D

Info on File versions:

model_*.py makes a specific model for radmc to do RT and images
    model_ulrich_disk_1: as given inside ysomodels.py in original repo
                     _2: uses CH3OH


make_[molecule]_line.py uses radmc and constucts the [molecule] line cube

yso_models.py defines multiple model_ as functions, to be used if trying different parameters.

