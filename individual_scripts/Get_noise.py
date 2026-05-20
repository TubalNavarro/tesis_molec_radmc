import numpy as np
from astropy.io import fits
from astropy.stats import mad_std


# =========================
# Parámetros
# =========================

cube_noise_reference = "../../cubes/DIHCA_cubes/shared_data/G335.78/G355.78+0.17_2_subcube.fits"
output_noise_file = "../../tesis_molec_radmc/inputs/G328_noise.dat"

# Si tienes cubo observacional, usa:
use_observational_cube = False

# Si NO tienes cubo observacional, usa ruido constante:
constant_rms = 2.89e-3   # valor del rms
nchan_constant = 56   # número de canales que quieres guardar


# =========================
# Funciones
# =========================

def make_noise_from_cube(cube_file, output_file):
    """
    Calcula el ruido por canal usando mad_std sobre cada plano espacial
    de un cubo observacional 3D con forma (canal, y, x).
    """

    with fits.open(cube_file) as hdul:
        data = hdul[0].data.astype(float)

    if data.ndim != 3:
        raise ValueError("El cubo debe ser 3D con forma (canal, y, x)")

    nchan = data.shape[0]
    noise_per_channel = np.zeros(nchan)

    for i in range(nchan):
        channel = data[i, :, :]
        noise_per_channel[i] = mad_std(channel, ignore_nan=True)

    np.savetxt(output_file, noise_per_channel)

    print(f"Ruido por canal guardado en: {output_file}")
    print(f"Número de canales: {nchan}")


def make_constant_noise(output_file, rms, nchan):
    """
    Genera un archivo .dat con ruido RMS constante para todos los canales.
    """

    noise_per_channel = np.full(nchan, rms)

    np.savetxt(output_file, noise_per_channel)

    print(f"Ruido constante guardado en: {output_file}")
    print(f"RMS constante: {rms}")
    print(f"Número de canales: {nchan}")


# =========================
# Ejecutar
# =========================

if use_observational_cube:
    make_noise_from_cube(
        cube_file=cube_noise_reference,
        output_file=output_noise_file
    )
else:
    make_constant_noise(
        output_file=output_noise_file,
        rms=constant_rms,
        nchan=nchan_constant
    )