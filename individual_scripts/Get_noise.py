import numpy as np
from astropy.io import fits
from astropy.stats import mad_std


# =========================
# Parámetros
# =========================

cube_noise_reference = "../../cubes/DIHCA_cubes/shared_data/G335.78/G355.78+0.17_2_subcube.fits"
output_noise_file = "../../tesis_molec_radmc/inputs/G355.78+0.17_2_noise.dat"


# =========================
# Leer cubo
# =========================

with fits.open(cube_noise_reference) as hdul:
    data = hdul[0].data.astype(float)

if data.ndim != 3:
    raise ValueError("El cubo debe ser 3D (canal, y, x)")


# =========================
# Calcular ruido
# =========================

nchan = data.shape[0]
noise_per_channel = np.zeros(nchan)

for i in range(nchan):
    channel = data[i, :, :]
    noise_per_channel[i] = mad_std(channel, ignore_nan=True)


# =========================
# Guardar a .dat
# =========================

np.savetxt(output_noise_file, noise_per_channel)

print(f"Ruido guardado en: {output_noise_file}")