from spectral_cube import SpectralCube

cube_folder="/home/tubal/repo_tesis/cubes/DIHCA_cubes/shared_data/G335.78/"

cube = SpectralCube.read(cube_folder+"spw0_from_CH3OH.structure15.subcube.fits")

subcube = cube[564:596, :, :]

print(cube.shape)
print(subcube.shape)

subcube.write(cube_folder+"G355.78+0.17_2_subcube.fits", overwrite=True)