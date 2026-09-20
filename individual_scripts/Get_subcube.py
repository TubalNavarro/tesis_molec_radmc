from spectral_cube import SpectralCube

cube_folder="/share/Part2/tubal/tesis/cubes/DIHCA_cubes/shared_data/G333.12/"

cube = SpectralCube.read(cube_folder+"G333.12-0.56_1.fits")

subcube = cube[562:620, :, :]

print(cube.shape)
print(subcube.shape)

subcube.write(cube_folder+"G333.12-0.56_1_subcube.fits", overwrite=True)