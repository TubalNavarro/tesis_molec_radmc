from astropy.io import fits
import numpy as np

# =========================
# Archivos
# =========================
input_fits = "pv/pv_G328.fits"
output_fits = "pv/pv_G328_flipped.fits"

# =========================
# Leer FITS
# =========================
with fits.open(input_fits) as hdul:
    hdu = hdul[0]
    data = hdu.data
    header = hdu.header.copy()

# =========================
# Revisar dimensiones
# =========================
if header["NAXIS"] != 2:
    raise ValueError("Este script está pensado para un FITS 2D tipo PV.")

ny = header["NAXIS2"]

# =========================
# Guardar valores originales
# =========================
crpix2_old = header["CRPIX2"]
crval2_old = header["CRVAL2"]
cdelt2_old = header["CDELT2"]

if cdelt2_old > 0:
    print("Advertencia: CDELT2 ya era positivo.")

# =========================
# Invertir datos en eje espectral
# =========================
# Para un FITS 2D leído por astropy:
# data.shape = (NAXIS2, NAXIS1)
data_flipped = data[::-1, :]

# =========================
# Nuevo WCS espectral
# =========================
cdelt2_new = -cdelt2_old  # ahora positivo

# Queremos que la nueva fila inferior tenga la velocidad
# que antes tenía la fila superior.
#
# Fórmula FITS:
# v = CRVAL2 + (p - CRPIX2) * CDELT2
#
# Pixel FITS inferior: p = 1
# Pixel FITS superior: p = ny

v_top_old = crval2_old + (ny - crpix2_old) * cdelt2_old

crval2_new = v_top_old - (1 - crpix2_old) * cdelt2_new

# Actualizar header
header["CDELT2"] = cdelt2_new
header["CRVAL2"] = crval2_new

header.add_history("Data flipped along spectral axis.")
header.add_history("CDELT2 sign changed and CRVAL2 adjusted consistently.")

# =========================
# Guardar nuevo FITS
# =========================
fits.writeto(output_fits, data_flipped, header, overwrite=True)

print(f"Guardado: {output_fits}")

# =========================
# Verificación
# =========================
def vel_from_header(hdr, p):
    return hdr["CRVAL2"] + (p - hdr["CRPIX2"]) * hdr["CDELT2"]

print()
print("Header original:")
print(f"  CDELT2 = {cdelt2_old:.6f} m/s")
print(f"  CRVAL2 = {crval2_old:.6f} m/s")
print(f"  v fila inferior = {vel_from_header(hdu.header, 1)/1000:.3f} km/s")
print(f"  v fila superior = {vel_from_header(hdu.header, ny)/1000:.3f} km/s")

print()
print("Header nuevo:")
print(f"  CDELT2 = {header['CDELT2']:.6f} m/s")
print(f"  CRVAL2 = {header['CRVAL2']:.6f} m/s")
print(f"  v fila inferior = {vel_from_header(header, 1)/1000:.3f} km/s")
print(f"  v fila superior = {vel_from_header(header, ny)/1000:.3f} km/s")